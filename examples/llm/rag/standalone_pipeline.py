# Copyright (c) 2023-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time

import cudf

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.concat_df import concat_dataframes
from morpheus_llm.llm import LLMEngine
from morpheus_llm.llm.nodes.extracter_node import ExtracterNode
from morpheus_llm.llm.nodes.rag_node import RAGNode
from morpheus_llm.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus_llm.stages.llm.llm_engine_stage import LLMEngineStage

from ..common.utils import build_huggingface_embeddings
from ..common.utils import build_llm_service
from ..common.utils import build_milvus_service

logger = logging.getLogger(__name__)


def _build_engine(model_name: str, vdb_resource_name: str, llm_service: str, embedding_size: int):

    engine = LLMEngine()

    engine.add_node("extracter", node=ExtracterNode())

    prompt = """You are a helpful assistant. Given the following background information:\n
{% for c in contexts -%}
{% if c.title %}
Title: {{ c.title }}
{% endif %}
{% if c.summary %}
Summary: {{ c.summary }}
{% endif %}
{% if c.page_content %}
Text: {{ c.page_content }}
{% endif %}
{% if c.header %}
Header: {{ c.header }}
{% endif %}
{% if c.data %}
Data: {{ c.data }}
{% endif %}
{% endfor %}

Please answer the following question: \n{{ query }}"""

    vector_service = build_milvus_service(embedding_size)
    embeddings = build_huggingface_embeddings("sentence-transformers/all-MiniLM-L6-v2",
                                              model_kwargs={'device': 'cuda'},
                                              encode_kwargs={'batch_size': 100})

    llm_service = build_llm_service(model_name, llm_service=llm_service, temperature=0.5, tokens_to_generate=200)

    # Async wrapper around embeddings
    async def calc_embeddings(texts: list[str]) -> list[list[float]]:
        return embeddings.embed_documents(texts)

    engine.add_node("rag",
                    inputs=["/extracter"],
                    node=RAGNode(prompt=prompt,
                                 vdb_service=vector_service.load_resource(vdb_resource_name),
                                 embedding=calc_embeddings,
                                 llm_client=llm_service))

    engine.add_task_handler(inputs=["/rag"], handler=SimpleTaskHandler())

    return engine


def standalone(num_threads,
               pipeline_batch_size,
               model_max_batch_size,
               model_name,
               vdb_resource_name,
               repeat_count,
               llm_service: str,
               embedding_size: int,
               question: list[str]):
    config = Config()
    config.mode = PipelineModes.NLP
    config.edge_buffer_size = 128

    # Below properties are specified by the command line
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size

    source_dfs = [cudf.DataFrame({"questions": question})]

    completion_task = {"task_type": "completion", "task_dict": {"input_keys": ["questions"], }}

    pipe = LinearPipeline(config)

    pipe.set_source(InMemorySourceStage(config, dataframes=source_dfs, repeat=repeat_count))

    pipe.add_stage(DeserializeStage(config, task_type="llm_engine", task_payload=completion_task))

    pipe.add_stage(MonitorStage(config, description="Source rate", unit='questions'))

    pipe.add_stage(
        LLMEngineStage(config,
                       engine=_build_engine(model_name=model_name,
                                            vdb_resource_name=vdb_resource_name,
                                            llm_service=llm_service,
                                            embedding_size=embedding_size)))

    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.add_stage(MonitorStage(config, description="Response rate", unit="responses", delayed_start=True))

    start_time = time.time()

    pipe.run()

    messages = sink.get_messages()
    responses = concat_dataframes(messages)
    logger.info("Pipeline complete. Received %s responses", len(responses))

    if logger.isEnabledFor(logging.DEBUG):
        # The responses are quite long, when debug is enabled disable the truncation that pandas and cudf normally
        # perform on the output
        # pd.set_option('display.max_colwidth', None)

        for row in responses.itertuples():
            logger.debug("Question:\n%s", row.questions)
            logger.debug("Response:\n%s", row.response)

    return start_time
