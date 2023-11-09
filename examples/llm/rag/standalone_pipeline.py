# Copyright (c) 2023, NVIDIA CORPORATION.
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
from morpheus.llm import LLMEngine
from morpheus.llm.nodes.extracter_node import ExtracterNode
from morpheus.llm.nodes.rag_node import RAGNode
from morpheus.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus.messages import ControlMessage
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.llm.llm_engine_stage import LLMEngineStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage

from ..common.utils import build_huggingface_embeddings
from ..common.utils import build_llm_service
from ..common.utils import build_milvus_service

logger = logging.getLogger(__name__)


def _build_engine(model_name: str, model_type: str, vdb_resource_name: str):

    engine = LLMEngine()

    engine.add_node("extracter", node=ExtracterNode())

    prompt = """You are a helpful assistant. Given the following background information:\n
{% for c in contexts -%}
Title: {{ c.title }}
Summary: {{ c.summary }}
Text: {{ c.page_content }}
{% endfor %}

Please answer the following question: \n{{ query }}"""

    vector_service = build_milvus_service(384)
    embeddings = build_huggingface_embeddings("sentence-transformers/all-MiniLM-L6-v2",
                                              model_kwargs={'device': 'cuda'},
                                              encode_kwargs={'batch_size': 100})
    llm_service = build_llm_service(model_name, model_type=model_type, temperature=0.5, tokens_to_generate=200)

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


def standalone(
        num_threads,
        pipeline_batch_size,
        model_max_batch_size,
        model_name,
        model_type,
        vdb_resource_name,
        repeat_count,
):
    # Configuration setup for the pipeline
    config = Config()
    config.mode = PipelineModes.OTHER  # Initial mode set to OTHER, will be overridden below

    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.mode = PipelineModes.NLP
    config.edge_buffer_size = 128  # Set edge buffer size for the pipeline stages

    # Create a DataFrame as the data source for the pipeline
    source_dfs = [
        cudf.DataFrame({"questions": ["What are some new attacks discovered in the cyber security industry?."] * 5})
    ]

    # Define a task to be used by the pipeline stages
    completion_task = {"task_type": "completion", "task_dict": {"input_keys": ["questions"], }}

    # Initialize the pipeline with the configuration
    pipe = LinearPipeline(config)

    # Set the source stage of the pipeline with the DataFrame and repeat count
    pipe.set_source(InMemorySourceStage(config, dataframes=source_dfs, repeat=repeat_count))

    # Add deserialization stage to convert messages for processing
    pipe.add_stage(
        DeserializeStage(config, message_type=ControlMessage, task_type="llm_engine", task_payload=completion_task))

    # Add a monitoring stage to observe the source data rate
    pipe.add_stage(MonitorStage(config, description="Source rate", unit='questions'))

    # Add the main LLM engine stage to the pipeline with the model and vector database
    pipe.add_stage(
        LLMEngineStage(config, engine=_build_engine(model_name=model_name, model_type=model_type, vdb_resource_name=vdb_resource_name)))

    # Add a sink stage to collect the output from the pipeline
    sink = pipe.add_stage(InMemorySinkStage(config))

    # Add another monitoring stage to observe the response rate with a delayed start
    pipe.add_stage(MonitorStage(config, description="Response rate", unit="responses", delayed_start=True))

    start_time = time.time()

    pipe.run()

    # Log the total number of responses received after pipeline completion
    logger.info("Pipeline complete. Received %s responses", len(sink.get_messages()))

    # Return the start time for performance measurement or further processing
    return start_time
