# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmark the examples/llm/rag/standalone_pipeline.py example"""

import collections.abc
import os
import types
import typing

import pytest

import cudf

from _utils import TEST_DIRS
from _utils.dataset_manager import DatasetManager
from _utils.milvus import populate_milvus
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.llm import LLMEngine
from morpheus.llm.nodes.extracter_node import ExtracterNode
from morpheus.llm.nodes.rag_node import RAGNode
from morpheus.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus.messages import ControlMessage
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.llm.llm_engine_stage import LLMEngineStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.service.vdb.milvus_vector_db_service import MilvusVectorDBService

EMBEDDING_SIZE = 384
QUESTION = "What are some new attacks discovered in the cyber security industry?"
PROMPT = """You are a helpful assistant. Given the following background information:\n
{% for c in contexts -%}
Title: {{ c.title }}
Summary: {{ c.summary }}
Text: {{ c.page_content }}
{% endfor %}

Please answer the following question: \n{{ query }}"""
EXPECTED_RESPONSE = "Ransomware, Phishing, Malware, Denial of Service, SQL injection, and Password Attacks"


def _build_engine(milvus_service: MilvusVectorDBService,
                  llm_service_name: str,
                  model_name: str,
                  collection_name: str,
                  utils_mod: types.ModuleType):
    engine = LLMEngine()
    engine.add_node("extracter", node=ExtracterNode())

    embeddings = utils_mod.build_huggingface_embeddings("sentence-transformers/all-MiniLM-L6-v2",
                                                        model_kwargs={'device': 'cuda'},
                                                        encode_kwargs={'batch_size': 100})

    llm_service = utils_mod.build_llm_service(model_name=model_name,
                                              llm_service=llm_service_name,
                                              temperature=0.5,
                                              tokens_to_generate=200)

    # Async wrapper around embeddings
    async def calc_embeddings(texts: list[str]) -> list[list[float]]:
        return embeddings.embed_documents(texts)

    engine.add_node("rag",
                    inputs=["/extracter"],
                    node=RAGNode(prompt=PROMPT,
                                 vdb_provider=milvus_service.load_resource(collection_name),
                                 embedding=calc_embeddings,
                                 llm_client=llm_service))

    engine.add_task_handler(inputs=["/rag"], handler=SimpleTaskHandler())

    return engine


def _run_pipeline(config: Config,
                  llm_service_name: str,
                  model_name: str,
                  milvus_server_uri: str,
                  collection_name: str,
                  repeat_count: int,
                  utils_mod: types.ModuleType):

    config.mode = PipelineModes.NLP
    config.edge_buffer_size = 128
    config.pipeline_batch_size = 1024
    config.model_max_batch_size = 64

    questions = [QUESTION] * repeat_count
    source_df = cudf.DataFrame({"questions": questions})

    completion_task = {"task_type": "completion", "task_dict": {"input_keys": ["questions"], }}
    pipe = LinearPipeline(config)

    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))

    pipe.add_stage(
        DeserializeStage(config, message_type=ControlMessage, task_type="llm_engine", task_payload=completion_task))
    
    milvus_service = utils_mod.build_milvus_service(embedding_size=EMBEDDING_SIZE, uri=milvus_server_uri)

    pipe.add_stage(
        LLMEngineStage(config,
                       engine=_build_engine(milvus_service=milvus_service,
                                            llm_service_name=llm_service_name,
                                            model_name=model_name,
                                            collection_name=collection_name,
                                            utils_mod=utils_mod)))

    pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    milvus_service.close()


@pytest.mark.milvus
@pytest.mark.use_python
@pytest.mark.use_cudf
@pytest.mark.benchmark
@pytest.mark.import_mod(os.path.join(TEST_DIRS.examples_dir, 'llm/common/utils.py'))
@pytest.mark.usefixtures("mock_nemollm", "mock_chat_completion")
@pytest.mark.parametrize("llm_service_name", ["nemollm", "openai"])
@pytest.mark.parametrize("repeat_count", [10, 100])
def test_rag_standalone_pipe(benchmark: collections.abc.Callable[[collections.abc.Callable], typing.Any],
                             config: Config,
                             dataset: DatasetManager,
                             milvus_server_uri: str,
                             repeat_count: int,
                             import_mod: types.ModuleType,
                             llm_service_name: str):
    collection_name = f"test_bench_rag_standalone_pipe_{llm_service_name}"
    populate_milvus(milvus_server_uri=milvus_server_uri,
                    collection_name=collection_name,
                    resource_kwargs=import_mod.build_milvus_config(embedding_size=EMBEDDING_SIZE),
                    df=dataset["service/milvus_rss_data.json"],
                    overwrite=True)

    benchmark(
        _run_pipeline,
        config=config,
        llm_service_name=llm_service_name,
        model_name="test_model",
        milvus_server_uri=milvus_server_uri,
        collection_name=collection_name,
        repeat_count=repeat_count,
        utils_mod=import_mod,
    )
