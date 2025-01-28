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

import pytest

from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.column_info import ColumnInfo
from morpheus.utils.column_info import CustomColumn
from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.logger import set_log_level


def test_constructor(config: Config):
    from morpheus_dfp.stages.dfp_preprocessing_stage import DFPPreprocessingStage

    schema = DataFrameInputSchema()
    stage = DFPPreprocessingStage(config, input_schema=schema)
    assert isinstance(stage, SinglePortStage)
    assert stage._input_schema is schema


@pytest.mark.usefixtures("reset_loglevel")
@pytest.mark.parametrize('log_level', [logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG])
def test_process_features(
        config: Config,
        control_message: "ControlMessage",  # noqa: F821
        dataset_pandas: DatasetManager,
        log_level: int):
    from morpheus_dfp.stages.dfp_preprocessing_stage import DFPPreprocessingStage

    set_log_level(log_level)

    expected_df = control_message.payload().copy_dataframe()
    expected_df['v210'] = expected_df['v2'] + 10
    expected_df['v3'] = expected_df['v3'].astype(str)

    schema = DataFrameInputSchema(column_info=[
        CustomColumn(name='v210', dtype=str, process_column_fn=lambda df: df['v2'] + 10),
        ColumnInfo(name='v3', dtype=str)
    ])

    stage = DFPPreprocessingStage(config, input_schema=schema)
    results = stage.process_features(control_message)

    assert isinstance(results, ControlMessage)
    dataset_pandas.assert_compare_df(results.payload().get_data(), expected_df)
