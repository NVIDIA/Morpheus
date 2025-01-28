#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import random
import types

import pytest

import cudf

from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.modules import to_control_message  # noqa: F401 # pylint: disable=unused-import
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_ids import TO_CONTROL_MESSAGE
from morpheus_llm.service.vdb.milvus_vector_db_service import MilvusVectorDBService
from morpheus_llm.stages.output.write_to_vector_db_stage import WriteToVectorDBStage


@pytest.fixture(name="pymilvus", scope='session', autouse=True)
def pymilvus_fixture(pymilvus: types.ModuleType):
    """
    Fixture to ensure pymilvus is installed
    """
    yield pymilvus


def get_test_df(num_input_rows):
    df = cudf.DataFrame({
        "id": list(range(num_input_rows)),
        "age": [random.randint(20, 40) for i in range(num_input_rows)],
        "embedding": [[random.random() for _ in range(3)] for _ in range(num_input_rows)]
    })

    return df


@pytest.mark.milvus
@pytest.mark.gpu_mode
@pytest.mark.parametrize("use_instance, num_input_rows, expected_num_output_rows, resource_kwargs, recreate",
                         [(True, 5, 5, {
                             "partition_name": "age_partition"
                         }, True), (False, 5, 5, {}, False), (False, 5, 5, {}, True)])
def test_write_to_vector_db_stage_from_cm_pipe(milvus_server_uri: str,
                                               idx_part_collection_config: dict,
                                               use_instance: bool,
                                               config: Config,
                                               num_input_rows: int,
                                               expected_num_output_rows: int,
                                               resource_kwargs: dict,
                                               recreate: bool):
    collection_name = "test_stage_cm_insert_collection"

    df = get_test_df(num_input_rows)

    milvus_service = MilvusVectorDBService(uri=milvus_server_uri)

    # Make sure to drop any existing collection from previous runs.
    milvus_service.drop(collection_name)
    # Create milvus collection using config file.
    milvus_service.create(name=collection_name, overwrite=True, **idx_part_collection_config)

    if recreate:
        # Update resource kwargs with collection configuration if recreate is True
        resource_kwargs.update(idx_part_collection_config)

    to_cm_module_config = {
        "module_id": TO_CONTROL_MESSAGE, "module_name": "to_control_message", "namespace": MORPHEUS_MODULE_NAMESPACE
    }

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [df]))
    pipe.add_stage(
        LinearModulesStage(config,
                           to_cm_module_config,
                           input_port_name="input",
                           output_port_name="output",
                           output_type=ControlMessage))

    # Provide partition name in the resource_kwargs to insert data into the partition
    # otherwise goes to '_default' partition.
    if use_instance:
        # Instantiate stage with service instance and insert options.
        write_to_vdb_stage = WriteToVectorDBStage(config,
                                                  resource_name=collection_name,
                                                  service=milvus_service,
                                                  recreate=recreate,
                                                  resource_kwargs=resource_kwargs)
    else:
        service_kwargs = {"uri": milvus_server_uri}
        # Instantiate stage with service name, uri and insert options.
        write_to_vdb_stage = WriteToVectorDBStage(config,
                                                  resource_name=collection_name,
                                                  service="milvus",
                                                  recreate=recreate,
                                                  resource_kwargs=resource_kwargs,
                                                  **service_kwargs)

    pipe.add_stage(write_to_vdb_stage)
    sink_stage = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    messages = sink_stage.get_messages()

    assert len(messages) == 1
    assert isinstance(messages[0], ControlMessage)
    assert messages[0].has_metadata("insert_response")

    # Insert entities response as a dictionary.
    response = messages[0].get_metadata("insert_response")

    status = response["status"]
    assert status in ["inserted", "accumulated"]

    if (status == "inserted"):
        assert response["insert_count"] == expected_num_output_rows
        assert response["succ_count"] == expected_num_output_rows
    else:
        assert response["accum_count"] == expected_num_output_rows

    assert response["err_count"] == 0


@pytest.mark.milvus
@pytest.mark.use_cpp
def test_no_resource_error(milvus_server_uri: str, config: Config):
    """
    Test to verify that the stage re-raises errors from the underlying service.
    """
    collection_name = "does_not_exist_collection"

    df = get_test_df(5)

    milvus_service = MilvusVectorDBService(uri=milvus_server_uri)

    # Make sure to drop any existing collection from previous runs.
    milvus_service.drop(collection_name)

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [df]))
    pipe.add_stage(DeserializeStage(config))

    # Instantiate stage with service instance and insert options.
    pipe.add_stage(WriteToVectorDBStage(config, resource_name=collection_name, service=milvus_service))

    with pytest.raises(ValueError):
        pipe.run()
