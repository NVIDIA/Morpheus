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
import types

import pytest

from _utils import TEST_DIRS
from _utils import assert_results
from morpheus.config import Config
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils.type_aliases import DataFrameType


def _check_pass_thru(config: Config, filter_probs_df: DataFrameType, pass_thru_stage_cls: SinglePortStage):
    pass_thru_stage = pass_thru_stage_cls(config)
    assert isinstance(pass_thru_stage, SinglePortStage)

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[filter_probs_df.copy(deep=True)]))
    sink_1 = pipe.add_stage(InMemorySinkStage(config))
    pipe.add_stage(pass_thru_stage)
    sink_2 = pipe.add_stage(InMemorySinkStage(config))
    comp_stage = pipe.add_stage(CompareDataFrameStage(config, filter_probs_df.copy(deep=True)))
    pipe.run()

    assert_results(comp_stage.get_results())

    in_messages = sink_1.get_messages()
    assert len(in_messages) == 1
    out_messages = sink_2.get_messages()
    assert len(out_messages) == 1
    assert in_messages[0] is out_messages[0]


@pytest.mark.gpu_and_cpu_mode
@pytest.mark.import_mod(os.path.join(TEST_DIRS.examples_dir, 'developer_guide/1_simple_python_stage/pass_thru.py'))
def test_pass_thru_ex1(config: Config, filter_probs_df: DataFrameType, import_mod: types.ModuleType):
    pass_thru = import_mod
    _check_pass_thru(config, filter_probs_df, pass_thru.PassThruStage)


@pytest.mark.gpu_and_cpu_mode
@pytest.mark.import_mod(os.path.join(TEST_DIRS.examples_dir, 'developer_guide/1_simple_python_stage/pass_thru_deco.py'))
def test_pass_thru_ex1_deco(config: Config, filter_probs_df: DataFrameType, import_mod: types.ModuleType):
    pass_thru = import_mod
    _check_pass_thru(config, filter_probs_df, pass_thru.pass_thru_stage)


@pytest.mark.gpu_and_cpu_mode
@pytest.mark.import_mod(
    os.path.join(TEST_DIRS.examples_dir, 'developer_guide/3_simple_cpp_stage/src/simple_cpp_stage/pass_thru.py'))
def test_pass_thru_ex3(config: Config, filter_probs_df: DataFrameType, import_mod: types.ModuleType):
    pass_thru = import_mod
    _check_pass_thru(config, filter_probs_df, pass_thru.PassThruStage)
