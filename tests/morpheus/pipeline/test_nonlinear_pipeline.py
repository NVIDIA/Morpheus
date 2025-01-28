#!/usr/bin/env python
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

import pytest

from _utils import assert_results
from _utils.dataset_manager import DatasetManager
from _utils.stages.split_stage import SplitStage
from morpheus.config import Config
from morpheus.pipeline.pipeline import Pipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage


def test_forking_pipeline(config: Config, dataset_cudf: DatasetManager):
    filter_probs_df = dataset_cudf["filter_probs.csv"]
    compare_higher_df = filter_probs_df[filter_probs_df["v2"] >= 0.5]
    compare_lower_df = filter_probs_df[filter_probs_df["v2"] < 0.5]

    pipe = Pipeline(config)

    # Create the stages
    source = pipe.add_stage(InMemorySourceStage(config, [filter_probs_df]))

    split_stage = pipe.add_stage(SplitStage(config))

    comp_higher = pipe.add_stage(CompareDataFrameStage(config, compare_df=compare_higher_df))
    comp_lower = pipe.add_stage(CompareDataFrameStage(config, compare_df=compare_lower_df))

    # Create the edges
    pipe.add_edge(source, split_stage)
    pipe.add_edge(split_stage.output_ports[0], comp_higher)
    pipe.add_edge(split_stage.output_ports[1], comp_lower)

    pipe.run()

    # Get the results
    assert_results(comp_higher.get_results())
    assert_results(comp_lower.get_results())


@pytest.mark.parametrize("source_count, expected_count", [(1, 1), (2, 2), (3, 3)])
def test_port_multi_sender(config: Config, dataset_cudf: DatasetManager, source_count: int, expected_count: int):

    filter_probs_df = dataset_cudf["filter_probs.csv"]

    pipe = Pipeline(config)

    input_ports = []
    for x in range(source_count):
        input_port = f"input_{x}"
        input_ports.append(input_port)

    sink_stage = pipe.add_stage(InMemorySinkStage(config))

    for x in range(source_count):
        source_stage = pipe.add_stage(InMemorySourceStage(config, [filter_probs_df]))
        pipe.add_edge(source_stage, sink_stage)

    pipe.run()

    assert len(sink_stage.get_messages()) == expected_count
