#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cudf

from _utils import TEST_DIRS
from _utils import assert_results
from morpheus.config import Config
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.preprocess.group_by_column_stage import GroupByColumnStage
from morpheus.utils.compare_df import compare_df
from morpheus.utils.type_aliases import DataFrameType


@pytest.fixture(name="_test_df", scope="module")
def _test_df_fixture():
    """
    Read the source data only once
    """
    # Manually reading this in since we need lines=False
    yield read_file_to_df(os.path.join(TEST_DIRS.tests_data_dir, 'azure_ad_logs.json'),
                          parser_kwargs={'lines': False},
                          df_type='pandas')


@pytest.fixture(name="test_df")
def test_df_fixture(_test_df: DataFrameType):
    """
    Ensure each test gets a unique copy
    """
    yield _test_df.copy(deep=True)


@pytest.mark.parametrize("group_by_column", ["identity", "location"])
def test_group_by_column_stage_pipe(config: Config, group_by_column: str, test_df: DataFrameType):
    input_df = cudf.from_pandas(test_df)
    input_df.drop(columns=["properties"], inplace=True)  # Remove once #1527 is resolved

    # Intentionally constructing the expected data in a manual way not involving pandas or cudf to avoid using the same
    # technology as the GroupByColumnStage
    rows = test_df.to_dict(orient="records")
    expected_data: dict[str, list[dict]] = {}
    for row in rows:
        key = row[group_by_column]
        if key not in expected_data:
            expected_data[key] = []

        row.pop('properties')  # Remove once #1527 is resolved
        expected_data[key].append(row)

    expected_dataframes: list[DataFrameType] = []
    for key in sorted(expected_data.keys()):
        df = pd.DataFrame(expected_data[key])
        expected_dataframes.append(df)

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[input_df]))
    pipe.add_stage(GroupByColumnStage(config, column_name=group_by_column))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages: MessageMeta = sink.get_messages()
    assert len(messages) == len(expected_dataframes)
    for (i, message) in enumerate(messages):
        output_df = message.copy_dataframe().to_pandas()
        output_df.reset_index(drop=True, inplace=True)

        expected_df = expected_dataframes[i]

        assert_results(compare_df(expected_df, output_df))
