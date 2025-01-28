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

import os
from collections import OrderedDict
from collections import namedtuple
from unittest import mock

# This import is needed to ensure our mocks work correctly, otherwise mlflow.pytorch is a stub.
import mlflow.pytorch  # noqa: F401 pylint: disable=unused-import
import pandas as pd
import pytest

from _utils import TEST_DIRS
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage

MockedRequests = namedtuple("MockedRequests", ["get", "patch", "response"])
MockedMLFlow = namedtuple("MockedMLFlow",
                          [
                              'MlflowClient',
                              'ModelSignature',
                              'RunsArtifactRepository',
                              'end_run',
                              'get_tracking_uri',
                              'log_metrics',
                              'log_params',
                              'model_info',
                              'model_src',
                              'pytorch_log_model',
                              'set_experiment',
                              'start_run'
                          ])


@pytest.fixture(name="databricks_env")
def databricks_env_fixture(restore_environ):  # pylint: disable=unused-argument
    env = {'DATABRICKS_HOST': 'https://test_host', 'DATABRICKS_TOKEN': 'test_token'}
    os.environ.update(env)
    yield env


@pytest.fixture(name="mock_requests")
def mock_requests_fixture():
    with mock.patch("requests.get") as mock_requests_get, mock.patch("requests.patch") as mock_requests_patch:
        mock_response = mock.MagicMock(status_code=200)
        mock_response.json.return_value = {'registered_model_databricks': {'id': 'test_id'}}
        mock_requests_get.return_value = mock_response
        yield MockedRequests(mock_requests_get, mock_requests_patch, mock_response)


@pytest.fixture
def mock_mlflow():
    with (mock.patch("morpheus.controllers.mlflow_model_writer_controller.MlflowClient") as mock_mlflow_client,
          mock.patch("morpheus.controllers.mlflow_model_writer_controller.ModelSignature") as mock_model_signature,
          mock.patch("morpheus.controllers.mlflow_model_writer_controller.RunsArtifactRepository") as
          mock_runs_artifact_repository,
          mock.patch("mlflow.end_run") as mock_mlflow_end_run,
          mock.patch("mlflow.get_tracking_uri") as mock_mlflow_get_tracking_uri,
          mock.patch("mlflow.log_metrics") as mock_mlflow_log_metrics,
          mock.patch("mlflow.log_params") as mock_mlflow_log_params,
          mock.patch("mlflow.pytorch.log_model") as mock_mlflow_pytorch_log_model,
          mock.patch("mlflow.set_experiment") as mock_mlflow_set_experiment,
          mock.patch("mlflow.start_run") as mock_mlflow_start_run):

        mock_mlflow_client.return_value = mock_mlflow_client
        mock_model_signature.return_value = mock_model_signature

        mock_model_info = mock.MagicMock()
        mock_mlflow_pytorch_log_model.return_value = mock_model_info

        mock_model_src = mock.MagicMock()
        mock_runs_artifact_repository.get_underlying_uri.return_value = mock_model_src

        mock_experiment = mock.MagicMock()
        mock_experiment.experiment_id = "test_experiment_id"
        mock_mlflow_set_experiment.return_value = mock_experiment

        mock_mlflow_start_run.return_value = mock_mlflow_start_run
        mock_mlflow_start_run.__enter__.return_value = mock_mlflow_start_run
        mock_mlflow_start_run.info.run_id = "test_run_id"
        mock_mlflow_start_run.info.run_uuid = "test_run_uuid"

        yield MockedMLFlow(mock_mlflow_client,
                           mock_model_signature,
                           mock_runs_artifact_repository,
                           mock_mlflow_end_run,
                           mock_mlflow_get_tracking_uri,
                           mock_mlflow_log_metrics,
                           mock_mlflow_log_params,
                           mock_model_info,
                           mock_model_src,
                           mock_mlflow_pytorch_log_model,
                           mock_mlflow_set_experiment,
                           mock_mlflow_start_run)


def test_constructor(config: Config):
    from morpheus_dfp.stages.dfp_mlflow_model_writer import DFPMLFlowModelWriterStage

    stage = DFPMLFlowModelWriterStage(config,
                                      model_name_formatter="test_model_name-{user_id}-{user_md5}",
                                      experiment_name_formatter="/test/{user_id}-{user_md5}-{reg_model_name}",
                                      databricks_permissions={'test': 'this'})
    assert isinstance(stage, SinglePortStage)
    assert stage._controller.model_name_formatter == "test_model_name-{user_id}-{user_md5}"
    assert stage._controller.experiment_name_formatter == "/test/{user_id}-{user_md5}-{reg_model_name}"
    assert stage._controller.databricks_permissions == {'test': 'this'}


@pytest.mark.parametrize(
    "model_name_formatter,user_id,expected_val",
    [("test_model_name-{user_id}", 'test_user', "test_model_name-test_user"),
     ("test_model_name-{user_id}-{user_md5}", 'test_user',
      "test_model_name-test_user-9da1f8e0aecc9d868bad115129706a77"),
     ("test_model_name-{user_id}", 'test_城安宮川', "test_model_name-test_城安宮川"),
     ("test_model_name-{user_id}-{user_md5}", 'test_城安宮川', "test_model_name-test_城安宮川-c9acc3dec97777c8b6fd8ae70a744ea8")
     ])
def test_user_id_to_model(config: Config, model_name_formatter: str, user_id: str, expected_val: str):
    from morpheus_dfp.stages.dfp_mlflow_model_writer import DFPMLFlowModelWriterStage

    stage = DFPMLFlowModelWriterStage(config, model_name_formatter=model_name_formatter)
    assert stage._controller.user_id_to_model(user_id) == expected_val


@pytest.mark.parametrize("experiment_name_formatter,user_id,expected_val",
                         [("/test/expr/{reg_model_name}", 'test_user', "/test/expr/dfp-test_user"),
                          ("/test/expr/{reg_model_name}-{user_id}", 'test_user', "/test/expr/dfp-test_user-test_user"),
                          ("/test/expr/{reg_model_name}-{user_id}-{user_md5}",
                           'test_user',
                           "/test/expr/dfp-test_user-test_user-9da1f8e0aecc9d868bad115129706a77"),
                          ("/test/expr/{reg_model_name}", 'test_城安宮川', "/test/expr/dfp-test_城安宮川"),
                          ("/test/expr/{reg_model_name}-{user_id}", 'test_城安宮川', "/test/expr/dfp-test_城安宮川-test_城安宮川"),
                          ("/test/expr/{reg_model_name}-{user_id}-{user_md5}",
                           'test_城安宮川',
                           "/test/expr/dfp-test_城安宮川-test_城安宮川-c9acc3dec97777c8b6fd8ae70a744ea8")])
def test_user_id_to_experiment(config: Config, experiment_name_formatter: str, user_id: str, expected_val: str):
    from morpheus_dfp.stages.dfp_mlflow_model_writer import DFPMLFlowModelWriterStage

    stage = DFPMLFlowModelWriterStage(config,
                                      model_name_formatter="dfp-{user_id}",
                                      experiment_name_formatter=experiment_name_formatter)
    assert stage._controller.user_id_to_experiment(user_id) == expected_val


def verify_apply_model_permissions(mock_requests: MockedRequests,
                                   databricks_env: dict,
                                   databricks_permissions: OrderedDict,
                                   experiment_name: str):
    expected_headers = {"Authorization": f"Bearer {databricks_env['DATABRICKS_TOKEN']}"}
    mock_requests.get.assert_called_once_with(
        url=f"{databricks_env['DATABRICKS_HOST']}/api/2.0/mlflow/databricks/registered-models/get",
        headers=expected_headers,
        params={"name": experiment_name},
        timeout=10)

    expected_acl = [{'group_name': group, 'permission_level': pl} for (group, pl) in databricks_permissions.items()]

    mock_requests.patch.assert_called_once_with(
        url=f"{databricks_env['DATABRICKS_HOST']}/api/2.0/preview/permissions/registered-models/test_id",
        headers=expected_headers,
        json={'access_control_list': expected_acl},
        timeout=10)


def test_apply_model_permissions(config: Config, databricks_env: dict, mock_requests: MockedRequests):
    from morpheus_dfp.stages.dfp_mlflow_model_writer import DFPMLFlowModelWriterStage
    databricks_permissions = OrderedDict([('group1', 'CAN_READ'), ('group2', 'CAN_WRITE')])
    stage = DFPMLFlowModelWriterStage(config, databricks_permissions=databricks_permissions, timeout=10)
    stage._controller._apply_model_permissions("test_experiment")

    verify_apply_model_permissions(mock_requests, databricks_env, databricks_permissions, 'test_experiment')


@pytest.mark.usefixtures("restore_environ")
@pytest.mark.parametrize("databricks_host,databricks_token", [
    ("test_host", None),
    (None, "test_token"),
    (None, None),
])
def test_apply_model_permissions_no_perms_error(config: Config,
                                                databricks_host: str,
                                                databricks_token: str,
                                                mock_requests: MockedRequests):
    if databricks_host is not None:
        os.environ["DATABRICKS_HOST"] = databricks_host
    else:
        os.environ.pop("DATABRICKS_HOST", None)

    if databricks_token is not None:
        os.environ["DATABRICKS_TOKEN"] = databricks_token
    else:
        os.environ.pop("DATABRICKS_TOKEN", None)

    from morpheus_dfp.stages.dfp_mlflow_model_writer import DFPMLFlowModelWriterStage
    stage = DFPMLFlowModelWriterStage(config)
    with pytest.raises(RuntimeError):
        stage._controller._apply_model_permissions("test_experiment")

    mock_requests.get.assert_not_called()
    mock_requests.patch.assert_not_called()


@pytest.mark.usefixtures("databricks_env")
def test_apply_model_permissions_requests_error(config: Config, mock_requests: MockedRequests):
    from morpheus_dfp.stages.dfp_mlflow_model_writer import DFPMLFlowModelWriterStage
    mock_requests.get.side_effect = RuntimeError("test error")

    stage = DFPMLFlowModelWriterStage(config, timeout=10)
    stage._controller._apply_model_permissions("test_experiment")

    # This method just catches and logs any errors
    mock_requests.get.assert_called_once()
    mock_requests.patch.assert_not_called()


@pytest.mark.parametrize("databricks_permissions", [None, {}])
@pytest.mark.parametrize("tracking_uri", ['file:///home/user/morpheus/mlruns', "databricks"])
def test_on_data(
        config: Config,
        mock_mlflow: MockedMLFlow,  # pylint: disable=redefined-outer-name
        mock_requests: MockedRequests,
        dataset_pandas: DatasetManager,
        databricks_env: dict,
        databricks_permissions: dict,
        tracking_uri: str):
    from morpheus_dfp.stages.dfp_mlflow_model_writer import DFPMLFlowModelWriterStage
    from morpheus_dfp.stages.dfp_mlflow_model_writer import conda_env

    should_apply_permissions = (databricks_permissions is not None and tracking_uri == "databricks")

    if not should_apply_permissions:
        # We aren't setting databricks_permissions, so we shouldn't be trying to make any request calls
        mock_requests.get.side_effect = RuntimeError("should not be called")
        mock_requests.patch.side_effect = RuntimeError("should not be called")

    mock_mlflow.get_tracking_uri.return_value = tracking_uri

    config.ae.timestamp_column_name = 'eventTime'

    input_file = os.path.join(TEST_DIRS.validation_data_dir, "dfp-cloudtrail-role-g-validation-data-input.csv")
    df = dataset_pandas[input_file]
    time_col = df['eventTime']
    min_time = time_col.min()
    max_time = time_col.max()

    mock_model = mock.MagicMock()
    mock_model.learning_rate_decay.state_dict.return_value = {'last_epoch': 42}
    mock_model.learning_rate = 0.1
    mock_model.batch_size = 100

    mock_embedding = mock.MagicMock()
    mock_embedding.num_embeddings = 101
    mock_embedding.embedding_dim = 102
    mock_model.categorical_fts = {'test': {'embedding': mock_embedding}}

    mock_model.prepare_df.return_value = df
    mock_model.get_anomaly_score.return_value = pd.Series(float(i) for i in range(len(df)))

    msg = ControlMessage()
    msg.payload(MessageMeta(df))
    msg.set_metadata("model", mock_model)
    msg.set_metadata("user_id", 'Account-123456789')

    stage = DFPMLFlowModelWriterStage(config, databricks_permissions=databricks_permissions, timeout=10)
    assert stage._controller.on_data(msg) is msg  # Should be a pass-thru

    # Test mocks in order that they're called
    mock_mlflow.end_run.assert_called_once()
    mock_mlflow.set_experiment.assert_called_once_with("/dfp-models/dfp-Account-123456789")
    mock_mlflow.start_run.assert_called_once_with(run_name="autoencoder model training run",
                                                  experiment_id="test_experiment_id")

    mock_mlflow.log_params.assert_called_once_with({
        "Algorithm": "Denosing Autoencoder",
        "Epochs": 42,
        "Learning rate": 0.1,
        "Batch size": 100,
        "Start Epoch": min_time,
        "End Epoch": max_time,
        "Log Count": len(df)
    })

    mock_mlflow.log_metrics.assert_called_once_with({
        "embedding-test-num_embeddings": 101, "embedding-test-embedding_dim": 102
    })

    mock_model.prepare_df.assert_called_once()
    mock_model.get_anomaly_score.assert_called_once()

    mock_mlflow.ModelSignature.assert_called_once()

    mock_mlflow.pytorch_log_model.assert_called_once_with(pytorch_model=mock_model,
                                                          artifact_path="dfencoder-test_run_uuid",
                                                          conda_env=conda_env,
                                                          signature=mock_mlflow.ModelSignature)

    mock_mlflow.MlflowClient.assert_called_once()
    mock_mlflow.MlflowClient.create_registered_model.assert_called_once_with("dfp-Account-123456789")

    if databricks_permissions is not None:
        mock_mlflow.get_tracking_uri.assert_called_once()
    else:
        mock_mlflow.get_tracking_uri.assert_not_called()

    if should_apply_permissions:
        verify_apply_model_permissions(mock_requests, databricks_env, databricks_permissions, 'dfp-Account-123456789')
    else:
        mock_requests.get.assert_not_called()
        mock_requests.patch.assert_not_called()

    mock_mlflow.RunsArtifactRepository.get_underlying_uri.assert_called_once_with(mock_mlflow.model_info.model_uri)

    expected_tags = {"start": min_time, "end": max_time, "count": len(df)}

    mock_mlflow.MlflowClient.create_model_version.assert_called_once_with(name="dfp-Account-123456789",
                                                                          source=mock_mlflow.model_src,
                                                                          run_id="test_run_id",
                                                                          tags=expected_tags)
