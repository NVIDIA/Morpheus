# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

import time

import mrc
import mrc.core.operators as ops
from mrc.core.node import Broadcast

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.llm import LLMEngine
from morpheus.llm.nodes.extracter_node import ExtracterNode
from morpheus.llm.nodes.rag_node import RAGNode
from morpheus.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.pipeline import Pipeline
from morpheus.pipeline.stage import Stage
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.service.vdb.vector_db_service import VectorDBResourceService
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.input.kafka_source_stage import KafkaSourceStage
from morpheus.stages.llm.llm_engine_stage import LLMEngineStage
from morpheus.stages.output.write_to_kafka_stage import WriteToKafkaStage
from morpheus.stages.output.write_to_vector_db_stage import WriteToVectorDBStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage

from ..common.utils import build_default_milvus_config
from ..common.utils import build_llm_service
from ..common.utils import build_milvus_service


class SplitStage(Stage):

    def __init__(self, c: Config):
        super().__init__(c)

        self._create_ports(1, 2)

    @property
    def name(self) -> str:
        return "split"

    def supports_cpp_node(self):
        return False

    def compute_schema(self, schema: StageSchema):
        schema.output_schemas[0].set_type(schema.input_type)
        schema.output_schemas[1].set_type(schema.input_type)

    def _build(self, builder: mrc.Builder, input_nodes: list[mrc.SegmentObject]) -> list[mrc.SegmentObject]:
        assert len(input_nodes) == 1, "Only 1 input supported"

        # Create a broadcast node
        broadcast = Broadcast(builder, "broadcast")
        builder.make_edge(input_nodes[0], broadcast)

        def filter_higher_fn(data: MessageMeta):
            return MessageMeta(data.df[data.df["v2"] >= 0.5])

        def filter_lower_fn(data: MessageMeta):
            return MessageMeta(data.df[data.df["v2"] < 0.5])

        # Create a node that only passes on rows >= 0.5
        filter_higher = builder.make_node("filter_higher", ops.map(filter_higher_fn))
        builder.make_edge(broadcast, filter_higher)

        # Create a node that only passes on rows < 0.5
        filter_lower = builder.make_node("filter_lower", ops.map(filter_lower_fn))
        builder.make_edge(broadcast, filter_lower)

        return [filter_higher, filter_lower]


def _build_engine(model_name: str, vdb_service: VectorDBResourceService, llm_service: str):
    engine = LLMEngine()

    engine.add_node("extracter", node=ExtracterNode())

    prompt = """You are a helpful assistant. Given the following background information:\n
{% for c in contexts -%}
Title: {{ c.title }}
Summary: {{ c.summary }}
Text: {{ c.page_content }}
{% endfor %}

Please answer the following question: \n{{ query }}"""

    llm_service = build_llm_service(model_name, llm_service=llm_service, temperature=0.5, tokens_to_generate=200)

    engine.add_node("rag",
                    inputs=[("/extracter/*", "*")],
                    node=RAGNode(prompt=prompt, vdb_service=vdb_service, embedding=None, llm_client=llm_service))

    engine.add_task_handler(inputs=["/rag"], handler=SimpleTaskHandler())

    return engine


def pipeline(num_threads, pipeline_batch_size, model_max_batch_size, embedding_size, model_name, llm_service: str):
    config = Config()
    config.mode = PipelineModes.OTHER

    # Below properties are specified by the command line
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.mode = PipelineModes.NLP
    config.edge_buffer_size = 128

    vdb_service = build_milvus_service(embedding_size=embedding_size)

    upload_task = {"task_type": "upload", "task_dict": {"input_keys": ["questions"], }}
    retrieve_task = {"task_type": "retrieve", "task_dict": {"input_keys": ["questions", "embedding"], }}

    pipe = Pipeline(config)

    # Source of the retrieval queries
    retrieve_source = pipe.add_stage(KafkaSourceStage(config, bootstrap_servers="auto", input_topic=["retrieve_input"]))

    retrieve_deserialize = pipe.add_stage(
        DeserializeStage(config, message_type=ControlMessage, task_type="llm_engine", task_payload=retrieve_task))

    pipe.add_edge(retrieve_source, retrieve_deserialize)

    # Source of continually uploading documents
    upload_source = pipe.add_stage(KafkaSourceStage(config, bootstrap_servers="auto", input_topic=["upload"]))

    upload_deserialize = pipe.add_stage(
        DeserializeStage(config, message_type=ControlMessage, task_type="llm_engine", task_payload=upload_task))

    pipe.add_edge(upload_source, upload_deserialize)

    # Join the sources into one for tokenization
    preprocess = pipe.add_stage(
        PreprocessNLPStage(config,
                           vocab_hash_file="data/bert-base-uncased-hash.txt",
                           do_lower_case=True,
                           truncation=True,
                           add_special_tokens=False,
                           column='content'))

    pipe.add_edge(upload_deserialize, preprocess)
    pipe.add_edge(retrieve_deserialize, preprocess)

    inference = pipe.add_stage(
        TritonInferenceStage(config,
                             model_name=model_name,
                             server_url="localhost:8001",
                             force_convert_inputs=True,
                             use_shared_memory=True))
    pipe.add_edge(preprocess, inference)

    # Split the results based on the task
    split = pipe.add_stage(SplitStage(config))
    pipe.add_edge(inference, split)

    # If it's a retrieve task, branch to the LLM engine for RAG
    retrieve_llm_engine = pipe.add_stage(
        LLMEngineStage(config,
                       engine=_build_engine(model_name=model_name,
                                            vdb_service=vdb_service.load_resource("RSS"),
                                            llm_service=llm_service)))
    pipe.add_edge(split.output_ports[0], retrieve_llm_engine)

    retrieve_results = pipe.add_stage(
        WriteToKafkaStage(config, bootstrap_servers="auto", output_topic="retrieve_output"))
    pipe.add_edge(retrieve_llm_engine, retrieve_results)

    # If it's an upload task, then send it to the database
    upload_vdb = pipe.add_stage(
        WriteToVectorDBStage(config,
                             resource_name="RSS",
                             resource_kwargs=build_default_milvus_config(embedding_size=embedding_size),
                             recreate=True,
                             service=vdb_service))
    pipe.add_edge(split.output_ports[1], upload_vdb)

    start_time = time.time()

    # Run the pipeline
    pipe.run()

    vdb_service.close()

    return start_time
