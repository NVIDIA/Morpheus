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

import pandas as pd
import pytest

from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.pipeline.single_port_stage import SinglePortStage

# pylint: disable=redefined-outer-name


@pytest.fixture(name="control_message")
def control_message_fixture(config: Config, control_message: "ControlMessage"):  # noqa F821
    # Fill in some values for columns that the stage is looking for
    with control_message.payload().mutable_dataframe() as df:
        step = (len(df) + 1) * 100
        df["mean_abs_z"] = list(range(0, len(df) * step, step))
        for (i, col) in enumerate(sorted(config.ae.feature_columns)):
            step = i + 1 * 100
            df[f"{col}_z_loss"] = list(range(0, len(df) * step, step))

    yield control_message


@pytest.fixture(name="expected_df")
def expected_df_fixture(config: Config, control_message: "ControlMessage"):  # noqa F821
    df = control_message.payload().copy_dataframe()
    df = df.to_pandas()
    expected_df = pd.DataFrame()
    expected_df["user"] = df[config.ae.userid_column_name]
    expected_df["time"] = df[config.ae.timestamp_column_name]
    expected_df["period"] = pd.to_datetime(df[config.ae.timestamp_column_name]).dt.to_period('min')
    expected_df["anomalyScore"] = df['mean_abs_z']
    for col in config.ae.feature_columns:
        expected_df[f"{col}_score"] = df[f"{col}_z_loss"]

    yield expected_df


def test_constructor(config: Config):
    from morpheus_dfp.stages.dfp_viz_postproc import DFPVizPostprocStage
    stage = DFPVizPostprocStage(config, period='M', output_dir='/fake/test/dir', output_prefix='test_prefix')

    assert isinstance(stage, SinglePortStage)
    assert stage._user_column_name == config.ae.userid_column_name
    assert stage._timestamp_column == config.ae.timestamp_column_name
    assert stage._feature_columns == config.ae.feature_columns
    assert stage._period == 'M'
    assert stage._output_dir == '/fake/test/dir'
    assert stage._output_prefix == 'test_prefix'
    assert not stage._output_filenames


def test_postprocess(
        config: Config,
        control_message: "ControlMessage",  # noqa: F821
        expected_df: pd.DataFrame,
        dataset_pandas: DatasetManager):
    from morpheus_dfp.stages.dfp_viz_postproc import DFPVizPostprocStage

    # _postprocess doesn't write to disk, so the fake output_dir, shouldn't be an issue
    stage = DFPVizPostprocStage(config, period='min', output_dir='/fake/test/dir', output_prefix='test_prefix')
    results = stage._postprocess(control_message)

    assert isinstance(results, pd.DataFrame)
    dataset_pandas.assert_compare_df(results, expected_df)


def test_write_to_files(
        config: Config,
        tmp_path: str,
        control_message: "ControlMessage",  # noqa: F821
        expected_df: pd.DataFrame,
        dataset_pandas: DatasetManager):
    from morpheus_dfp.stages.dfp_viz_postproc import DFPVizPostprocStage

    stage = DFPVizPostprocStage(config, period='min', output_dir=tmp_path, output_prefix='test_prefix_')
    assert stage._write_to_files(control_message) is control_message

    # The times in the DF have a 30 second step, so the number of unique minutes is half the length of the DF
    num_expected_periods = len(expected_df) // 2
    output_files = [os.path.join(tmp_path, f) for f in os.listdir(tmp_path)]

    expected_files = sorted(
        [os.path.join(tmp_path, f"test_prefix_2023-05-02 19:{minute:02d}.csv") for minute in range(8, 18)])

    assert sorted(stage._output_filenames) == expected_files
    assert len(output_files) == num_expected_periods
    assert sorted(output_files) == expected_files

    first_period = expected_df['period'].iloc[0]

    for i in range(num_expected_periods):
        minute = i + 8
        output_file = os.path.join(tmp_path, f"test_prefix_2023-05-02 19:{minute:02d}.csv")
        out_df = dataset_pandas.get_df(output_file, no_cache=True)

        period = first_period + i
        expected_period_df = expected_df[expected_df["period"] == period]
        expected_period_df = expected_period_df.drop(["period"], axis=1)
        expected_period_df.reset_index(drop=True, inplace=True)

        dataset_pandas.assert_compare_df(out_df, expected_period_df)
