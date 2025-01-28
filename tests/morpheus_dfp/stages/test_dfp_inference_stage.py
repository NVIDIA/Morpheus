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

import logging
from unittest import mock

import pandas as pd
import pytest

from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.logger import set_log_level

# pylint: disable=redefined-outer-name


@pytest.fixture(name="mock_mlflow_client", autouse=True)
def mock_mlflow_client_fixture():
    with mock.patch("morpheus_dfp.stages.dfp_inference_stage.MlflowClient") as mock_mlflow_client:
        mock_mlflow_client.return_value = mock_mlflow_client
        yield mock_mlflow_client


@pytest.fixture(name="mock_model_manager", autouse=True)
def mock_model_manager_fixture():
    with mock.patch("morpheus_dfp.stages.dfp_inference_stage.ModelManager") as mock_model_manager:
        mock_model_manager.return_value = mock_model_manager
        yield mock_model_manager


def test_constructor(config: Config, mock_mlflow_client: mock.MagicMock, mock_model_manager: mock.MagicMock):
    from morpheus_dfp.stages.dfp_inference_stage import DFPInferenceStage

    stage = DFPInferenceStage(config, model_name_formatter="test_model_name-{user_id}-{user_md5}")

    assert isinstance(stage, SinglePortStage)
    assert stage._client is mock_mlflow_client
    assert stage._fallback_user == config.ae.fallback_username
    assert not stage._model_cache
    assert stage._model_manager is mock_model_manager

    mock_mlflow_client.assert_called_once()
    mock_model_manager.assert_called_once_with(model_name_formatter="test_model_name-{user_id}-{user_md5}")


def test_get_model(config: Config, mock_mlflow_client: mock.MagicMock, mock_model_manager: mock.MagicMock):
    from morpheus_dfp.stages.dfp_inference_stage import DFPInferenceStage

    mock_model_cache = mock.MagicMock()
    mock_model_manager.load_user_model.return_value = mock_model_cache

    stage = DFPInferenceStage(config)
    assert stage.get_model("test_user") is mock_model_cache

    mock_model_manager.load_user_model.assert_called_once_with(mock_mlflow_client,
                                                               user_id="test_user",
                                                               fallback_user_ids=[config.ae.fallback_username])


@pytest.mark.usefixtures("reset_loglevel")
@pytest.mark.parametrize('log_level', [logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG])
def test_on_data(
        config: Config,
        mock_mlflow_client: mock.MagicMock,  # pylint: disable=unused-argument
        mock_model_manager: mock.MagicMock,
        control_message: "ControlMessage",  # noqa: F821
        log_level: int,
        dataset_pandas: DatasetManager):
    from morpheus_dfp.stages.dfp_inference_stage import DFPInferenceStage

    set_log_level(log_level)

    expected_results = list(range(1000, control_message.payload().count + 1000))

    expected_df = control_message.payload().copy_dataframe()
    expected_df["results"] = expected_results
    expected_df["model_version"] = "test_model_name:test_model_version"

    mock_model = mock.MagicMock()
    mock_model.get_results.return_value = pd.DataFrame({"results": expected_results})

    mock_model_cache = mock.MagicMock()
    mock_model_cache.load_model.return_value = mock_model
    mock_model_cache.reg_model_name = "test_model_name"
    mock_model_cache.reg_model_version = "test_model_version"

    mock_model_manager.load_user_model.return_value = mock_model_cache

    stage = DFPInferenceStage(config, model_name_formatter="test_model_name-{user_id}")
    results = stage.on_data(control_message)

    assert isinstance(results, ControlMessage)
    assert results.payload() is control_message.payload()
    assert results.payload().count == control_message.payload().count
    dataset_pandas.assert_compare_df(results.payload().get_data(), expected_df)


@pytest.mark.parametrize("raise_error", [True, False])
def test_on_data_get_model_error(
        config: Config,
        mock_mlflow_client: mock.MagicMock,  # pylint: disable=unused-argument
        mock_model_manager: mock.MagicMock,
        control_message: "ControlMessage",  # noqa: F821
        raise_error: bool):
    from morpheus_dfp.stages.dfp_inference_stage import DFPInferenceStage

    # There are two error conditions that can occur in get_model can return None or raise an error
    if raise_error:
        mock_model_manager.load_user_model.side_effect = RuntimeError("test error")
    else:
        mock_model_manager.load_user_model.return_value = None

    stage = DFPInferenceStage(config, model_name_formatter="test_model_name-{user_id}")
    assert stage.on_data(control_message) is None
