#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import imghdr
import os
import subprocess

import pytest

from _utils import TEST_DIRS
from _utils import assert_path_exists
from _utils.dataset_manager import DatasetManager
from _utils.stages.conv_msg import ConvMsg
from morpheus.cli.commands import RANKDIR_CHOICES
from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.pipeline import Pipeline
from morpheus.pipeline.pipeline import PipelineState
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.postprocess.add_classifications_stage import AddClassificationsStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage


@pytest.fixture(name="viz_pipeline", scope="function")
def viz_pipeline_fixture(config: Config, dataset_cudf: DatasetManager):
    """
    Creates a quick pipeline.
    """
    config.class_labels = ['frogs', 'lizards', 'toads', 'turtles']
    config.num_threads = 1

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [dataset_cudf["filter_probs.csv"]]))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(ConvMsg(config, dataset_cudf["filter_probs.csv"]))
    pipe.add_stage(AddClassificationsStage(config))
    pipe.add_stage(SerializeStage(config, include=[f"^{c}$" for c in config.class_labels]))
    pipe.add_stage(InMemorySinkStage(config))

    return pipe


def test_call_before_build(viz_pipeline: Pipeline, tmp_path: str):

    # Test is necessary to ensure run() is called first. See issue #230
    viz_file = os.path.join(tmp_path, 'pipeline.png')

    with pytest.raises(RuntimeError):
        viz_pipeline.visualize(viz_file, rankdir="LR")

    assert not os.path.exists(viz_file)


def test_viz_without_run(viz_pipeline: Pipeline, tmp_path: str):

    viz_file = os.path.join(tmp_path, 'pipeline.png')

    viz_pipeline.build()
    viz_pipeline.visualize(viz_file, rankdir="LR")

    # Verify that the output file exists and is a valid png file
    assert_path_exists(viz_file)
    assert imghdr.what(viz_file) == 'png'
    assert viz_pipeline.state != PipelineState.INITIALIZED


@pytest.mark.slow
@pytest.mark.parametrize("rankdir", RANKDIR_CHOICES)
def test_from_cli(tmp_path: str, dataset_pandas: DatasetManager, rankdir: str):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, 'filter_probs.csv')
    out_file = os.path.join(tmp_path, 'out.csv')
    viz_file = os.path.join(tmp_path, 'pipeline.png')
    cli = (f"morpheus run pipeline-other --viz_file={viz_file} --viz_direction={rankdir} "
           f"from-file --filename={input_file} to-file --filename={out_file}")
    subprocess.run(cli, check=True, shell=True)

    assert_path_exists(viz_file)
    assert imghdr.what(viz_file) == 'png'
    assert_path_exists(out_file)

    df = dataset_pandas.get_df(out_file, no_cache=True)
    dataset_pandas.assert_compare_df(df, dataset_pandas['filter_probs.csv'])
