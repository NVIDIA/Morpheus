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

import glob
import os
import types
import typing
from unittest import mock

import pandas as pd

from _utils import TEST_DIRS
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.pipeline.control_message_stage import ControlMessageStage
from morpheus.stages.input.appshield_source_stage import AppShieldSourceStage


class TestCreateFeaturesRWStage:
    # pylint: disable=no-name-in-module

    @mock.patch('stages.create_features.Client')
    def test_constructor(
            self,
            mock_dask_client,
            config: Config,
            dask_distributed: types.ModuleType,  # pylint: disable=unused-argument
            rwd_conf: dict,
            interested_plugins: typing.List[str]):
        mock_dask_client.return_value = mock_dask_client
        from common.data_models import FeatureConfig
        from common.feature_extractor import FeatureExtractor
        from stages.create_features import CreateFeaturesRWStage

        n_workers = 12
        threads_per_worker = 8
        stage = CreateFeaturesRWStage(config,
                                      interested_plugins=interested_plugins,
                                      feature_columns=rwd_conf['model_features'],
                                      file_extns=rwd_conf['file_extensions'],
                                      n_workers=n_workers,
                                      threads_per_worker=threads_per_worker)

        assert isinstance(stage, ControlMessageStage)
        assert stage._client is mock_dask_client
        scheduler_info = stage._client.scheduler_info()
        for worker in scheduler_info['workers'].values():
            assert worker['nthreads'] == threads_per_worker

        assert isinstance(stage._feature_config, FeatureConfig)
        assert stage._feature_config.file_extns == rwd_conf['file_extensions']
        assert stage._feature_config.interested_plugins == interested_plugins

        assert stage._feas_all_zeros == {c: 0 for c in rwd_conf['model_features']}

        assert isinstance(stage._fe, FeatureExtractor)
        assert stage._fe._config is stage._feature_config

    @mock.patch('stages.create_features.Client')
    def test_on_next(self,
                     mock_dask_client,
                     config: Config,
                     rwd_conf: dict,
                     interested_plugins: typing.List[str],
                     dataset_pandas: DatasetManager):
        from stages.create_features import CreateFeaturesRWStage

        test_data_dir = os.path.join(TEST_DIRS.tests_data_dir, 'examples/ransomware_detection')

        mock_dask_client.return_value = mock_dask_client
        mock_dask_client.map.return_value = mock.MagicMock()

        dask_results = dataset_pandas[os.path.join(test_data_dir, 'dask_results.csv')]

        mock_dask_future = mock.MagicMock()
        mock_dask_future.result.return_value = dask_results
        mock_dask_client.submit.return_value = mock_dask_future

        input_glob = os.path.join(TEST_DIRS.tests_data_dir, 'appshield', 'snapshot-1', '*.json')
        appshield_source_stage = AppShieldSourceStage(config,
                                                      input_glob,
                                                      plugins_include=interested_plugins,
                                                      cols_include=rwd_conf['raw_columns'],
                                                      cols_exclude=["SHA256"],
                                                      encoding='latin1')

        input_data = appshield_source_stage.files_to_dfs(glob.glob(input_glob),
                                                         cols_include=rwd_conf['raw_columns'],
                                                         cols_exclude=["SHA256"],
                                                         plugins_include=interested_plugins,
                                                         encoding='latin1')

        input_messages = appshield_source_stage._build_messages(input_data)

        # Make sure the input test date looks the way we expect it
        assert len(input_messages) == 1
        input_message = input_messages[0]
        assert input_message.get_metadata('source') == 'appshield'

        stage = CreateFeaturesRWStage(config,
                                      interested_plugins=interested_plugins,
                                      feature_columns=rwd_conf['model_features'],
                                      file_extns=rwd_conf['file_extensions'],
                                      n_workers=5,
                                      threads_per_worker=6)

        # make sure we have a mocked dask client
        assert stage._client is mock_dask_client

        messages = stage.on_next(input_message)

        dataframes = []
        for message in messages:
            assert message.get_metadata('source') == input_message.get_metadata('source')
            dataframes.append(message.payload().copy_dataframe().to_pandas())

        actual_df = pd.concat(dataframes, ignore_index=True)
        actual_df.sort_values(by=["pid_process", "snapshot_id"], inplace=True)
        actual_df.reset_index(drop=True, inplace=True)

        expected_df = dataset_pandas[os.path.join(test_data_dir, 'dask_results.csv')]
        expected_df['source_pid_process'] = 'appshield_' + expected_df.pid_process
        expected_df['ldrmodules_df_path'] = expected_df['ldrmodules_df_path'].astype(str)  # convert to string
        expected_df.sort_values(by=["pid_process", "snapshot_id"], inplace=True)
        expected_df.reset_index(drop=True, inplace=True)

        dataset_pandas.assert_compare_df(actual_df, expected_df)

    @mock.patch('stages.create_features.Client')
    def test_on_completed(self, mock_dask_client, config: Config, rwd_conf: dict, interested_plugins: typing.List[str]):
        from stages.create_features import CreateFeaturesRWStage
        mock_dask_client.return_value = mock_dask_client

        stage = CreateFeaturesRWStage(config,
                                      interested_plugins=interested_plugins,
                                      feature_columns=rwd_conf['model_features'],
                                      file_extns=rwd_conf['file_extensions'],
                                      n_workers=5,
                                      threads_per_worker=6)

        assert stage._client is mock_dask_client
        mock_dask_client.close.assert_not_called()

        stage.on_completed()

        mock_dask_client.close.assert_called_once()
