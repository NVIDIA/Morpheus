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
import pathlib

import pandas as pd
import pytest

from _utils import TEST_DIRS
from _utils import assert_results
from morpheus.common import FileTypes
from morpheus.common import determine_file_type
from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.io.deserializers import read_file_to_df
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage


@pytest.mark.slow
@pytest.mark.gpu_and_cpu_mode
@pytest.mark.parametrize("input_file",
                         [
                             os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv"),
                             os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.parquet"),
                             os.path.join(TEST_DIRS.tests_data_dir, 'examples/abp_pcap_detection/abp_pcap.jsonlines')
                         ],
                         ids=["csv", "parquet", "jsonlines"])
@pytest.mark.parametrize("filter_null", [False, True], ids=["no_filter", "filter_null"])
@pytest.mark.parametrize("use_pathlib", [False, True], ids=["no_pathlib", "pathlib"])
@pytest.mark.parametrize("repeat", [1, 2, 5], ids=["repeat1", "repeat2", "repeat5"])
def test_file_source_stage_pipe(config: Config, input_file: str, filter_null: bool, use_pathlib: bool, repeat: int):
    parser_kwargs = {}
    if determine_file_type(input_file) == FileTypes.JSON:
        # kwarg specific to pandas.read_json
        parser_kwargs['convert_dates'] = config.execution_mode == ExecutionMode.CPU

    expected_df = read_file_to_df(file_name=input_file,
                                  filter_nulls=filter_null,
                                  df_type="pandas",
                                  parser_kwargs=parser_kwargs)
    expected_df = pd.concat([expected_df for _ in range(repeat)])

    expected_df.reset_index(inplace=True)
    expected_df.drop('index', axis=1, inplace=True)

    if use_pathlib:
        input_file = pathlib.Path(input_file)

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file, repeat=repeat, filter_null=filter_null))
    comp_stage = pipe.add_stage(
        CompareDataFrameStage(config, compare_df=expected_df, exclude=["index"], reset_index=True))
    pipe.run()

    assert_results(comp_stage.get_results())
