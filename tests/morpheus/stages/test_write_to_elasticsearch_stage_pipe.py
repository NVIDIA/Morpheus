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

import typing
from unittest.mock import patch

import pytest
import yaml

from morpheus.config import Config
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.write_to_elasticsearch_stage import WriteToElasticsearchStage
from morpheus.utils.type_aliases import DataFrameType


def connection_kwargs_func(kwargs):
    kwargs["retry_on_status"] = 3
    kwargs["retry_on_timeout"] = 3 * 10

    return kwargs


@pytest.fixture(scope="function", name="connection_conf_file")
def connection_conf_file_fixture(tmp_path):
    connection_kwargs = {"hosts": [{"host": "localhost", "port": 9201, "scheme": "http"}]}

    connection_conf_file = tmp_path / "connection_kwargs_conf.yaml"
    with connection_conf_file.open(mode="w") as file:
        yaml.dump(connection_kwargs, file)

    yield connection_conf_file


@pytest.mark.parametrize("conf_file, exception", [("connection_conf.yaml", FileNotFoundError), (None, Exception)])
def test_constructor_invalid_conf_file(config: Config,
                                       conf_file: str,
                                       exception: typing.Union[Exception, FileNotFoundError]):
    with pytest.raises(exception):
        WriteToElasticsearchStage(config, index="t_index", connection_conf_file=conf_file)


@patch("morpheus.controllers.elasticsearch_controller.Elasticsearch")
def test_constructor_with_custom_func(config: Config, connection_conf_file: str):
    expected_connection_kwargs = {
        "hosts": [{
            "host": "localhost", "port": 9201, "scheme": "http"
        }], "retry_on_status": 3, "retry_on_timeout": 30
    }

    stage = WriteToElasticsearchStage(config,
                                      index="t_index",
                                      connection_conf_file=connection_conf_file,
                                      connection_kwargs_update_func=connection_kwargs_func)

    assert stage._controller._connection_kwargs == expected_connection_kwargs


@pytest.mark.use_cudf
@patch("morpheus.stages.output.write_to_elasticsearch_stage.ElasticsearchController")
def test_write_to_elasticsearch_stage_pipe(mock_controller: typing.Any,
                                           connection_conf_file: str,
                                           config: Config,
                                           filter_probs_df: DataFrameType):
    mock_df_to_parallel_bulk_write = mock_controller.return_value.df_to_parallel_bulk_write
    mock_refresh_client = mock_controller.return_value.refresh_client

    # Create a pipeline
    pipe = LinearPipeline(config)

    # Add the source stage and the WriteToElasticsearchStage to the pipeline
    pipe.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipe.add_stage(WriteToElasticsearchStage(config, index="t_index", connection_conf_file=connection_conf_file))

    # Run the pipeline
    pipe.run()

    expected_index = mock_df_to_parallel_bulk_write.call_args[1]["index"]
    actual_df = mock_df_to_parallel_bulk_write.call_args[1]["df"]

    mock_refresh_client.assert_called_once()
    mock_df_to_parallel_bulk_write.assert_called_once()

    assert expected_index == "t_index"
    assert actual_df.equals(filter_probs_df.to_pandas())
