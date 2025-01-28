# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest import mock

import pytest

import cudf

from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus_llm.llm import LLMEngine
from morpheus_llm.llm.nodes.extracter_node import ExtracterNode
from morpheus_llm.llm.nodes.retriever_node import RetrieverNode
from morpheus_llm.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus_llm.service.vdb.milvus_vector_db_service import MilvusVectorDBResourceService
from morpheus_llm.service.vdb.milvus_vector_db_service import MilvusVectorDBService
from morpheus_llm.stages.llm.llm_engine_stage import LLMEngineStage


def _build_engine(vdb_service, **similarity_search_kwargs) -> LLMEngine:
    mock_embedding = mock.AsyncMock(return_value=[[1.2, 2.3, 3.4], [4.5, 5.6, 6.7]])
    engine = LLMEngine()
    engine.add_node("extracter", node=ExtracterNode())
    engine.add_node("retriever",
                    inputs=["/extracter"],
                    node=RetrieverNode(service=vdb_service, embedding=mock_embedding, **similarity_search_kwargs))
    engine.add_task_handler(inputs=["/retriever"], handler=SimpleTaskHandler())

    return engine


def test_pipeline(config: Config):
    expected_output = [[1, 2, 3], [4, 5, 6]]

    values = {'prompt': ["prompt1", "prompt2"]}
    input_df = cudf.DataFrame(values)
    expected_df = input_df.copy(deep=True)
    expected_df["response"] = expected_output

    task_payload = {"task_type": "llm_engine", "task_dict": {"input_keys": sorted(values.keys())}}

    mock_vdb_service = mock.MagicMock()
    mock_vdb_service.similarity_search = mock.AsyncMock(return_value=[[1, 2, 3], [4, 5, 6]])

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[input_df]))
    pipe.add_stage(DeserializeStage(config, task_type="llm_engine", task_payload=task_payload))
    pipe.add_stage(LLMEngineStage(config, engine=_build_engine(vdb_service=mock_vdb_service)))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    message = sink.get_messages()[0]
    assert isinstance(message, ControlMessage)
    actual_df = message.payload().df

    # Using equals, as CompareDataFrameStage fails to compare.
    assert actual_df.to_pandas().equals(expected_df.to_pandas())


@pytest.mark.milvus
def test_pipeline_with_milvus(config: Config,
                              milvus_service: MilvusVectorDBService,
                              idx_part_collection_config: dict,
                              milvus_data: list[dict]):

    collection_name = "test_retriever_node_collection"
    # Make sure to drop any existing collection from previous runs.
    milvus_service.drop(collection_name)
    # Create a collection.
    milvus_service.create(collection_name, **idx_part_collection_config)
    resource_service: MilvusVectorDBResourceService = milvus_service.load_resource(name=collection_name)
    # Insert data into collection
    resource_service.insert(milvus_data)

    # Define a similarity_search filter.
    expr = "age==26 or age==27"

    values = {'prompt': ["prompt1", "prompt2"]}
    input_df = cudf.DataFrame(values)
    expected_df = input_df.copy(deep=True)
    expected_df["response"] = [[{
        'age': 27, 'id': 2
    }, {
        'age': 26, 'id': 1
    }], [{
        'age': 27, 'id': 2
    }, {
        'age': 26, 'id': 1
    }]]

    task_payload = {"task_type": "llm_engine", "task_dict": {"input_keys": sorted(values.keys())}}

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[input_df]))
    pipe.add_stage(DeserializeStage(config, task_type="llm_engine", task_payload=task_payload))
    pipe.add_stage(LLMEngineStage(config, engine=_build_engine(vdb_service=resource_service, expr=expr)))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    message = sink.get_messages()[0]
    assert isinstance(message, ControlMessage)
    actual_df = message.payload().df

    # Using equals, as CompareDataFrameStage fails to compare.
    assert actual_df.to_pandas().equals(expected_df.to_pandas())
