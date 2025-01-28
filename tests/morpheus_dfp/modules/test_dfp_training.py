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
from unittest import mock

import pytest

from _utils import TEST_DIRS
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage


def test_constructor(config: Config):
    from morpheus_dfp.stages.dfp_training import DFPTraining

    stage = DFPTraining(config, model_kwargs={'test': 'this'}, epochs=40, validation_size=0.5)
    assert isinstance(stage, SinglePortStage)
    assert stage._model_kwargs['test'] == 'this'
    assert stage._epochs == 40
    assert stage._validation_size == 0.5


@pytest.mark.parametrize('validation_size', [-1, -0.2, 1, 5])
def test_constructor_bad_validation_size(config: Config, validation_size: float):
    from morpheus_dfp.stages.dfp_training import DFPTraining

    with pytest.raises(ValueError):
        DFPTraining(config, validation_size=validation_size)


@pytest.mark.parametrize('validation_size', [0., 0.2])
@mock.patch('morpheus_dfp.stages.dfp_training.AutoEncoder')
@mock.patch('morpheus_dfp.stages.dfp_training.train_test_split')
def test_on_data(mock_train_test_split: mock.MagicMock,
                 mock_ae: mock.MagicMock,
                 config: Config,
                 dataset_pandas: DatasetManager,
                 validation_size: float):
    from morpheus_dfp.stages.dfp_training import DFPTraining

    mock_ae.return_value = mock_ae

    input_file = os.path.join(TEST_DIRS.validation_data_dir, "dfp-cloudtrail-role-g-validation-data-input.csv")
    df = dataset_pandas[input_file]
    train_df = df[df.columns.intersection(config.ae.feature_columns)]

    mock_validation_df = mock.MagicMock()
    mock_train_test_split.return_value = (train_df, mock_validation_df)

    msg = ControlMessage()
    msg.payload(MessageMeta(df))
    msg.set_metadata("user_id", 'Account-123456789')

    stage = DFPTraining(config, validation_size=validation_size)
    results = stage.on_data(msg)

    assert isinstance(results, ControlMessage)
    assert results.payload().count == msg.payload().count
    assert results.get_metadata("model") is mock_ae

    # Pandas doesn't like the comparison that mock will make if we called MagicMock.assert_called_once_with(df)
    # Checking the call args manually
    if validation_size > 0:
        expected_run_validation = True
        expected_val_data = mock_validation_df
        mock_train_test_split.assert_called_once()
        assert len(mock_train_test_split.call_args.args) == 1
        dataset_pandas.assert_compare_df(mock_train_test_split.call_args.args[0], train_df)
        assert mock_train_test_split.call_args.kwargs == {'test_size': validation_size, 'shuffle': False}
    else:
        expected_run_validation = False
        expected_val_data = None
        mock_train_test_split.assert_not_called()

    mock_ae.fit.assert_called_once()

    assert len(mock_ae.fit.call_args.args) == 1
    dataset_pandas.assert_compare_df(mock_ae.fit.call_args.args[0], train_df)
    assert mock_ae.fit.call_args.kwargs == {
        'epochs': stage._epochs, 'validation_data': expected_val_data, 'run_validation': expected_run_validation
    }

    # The stage shouldn't be modifying the dataframe
    dataset_pandas.assert_compare_df(results.payload().df, dataset_pandas[input_file])
