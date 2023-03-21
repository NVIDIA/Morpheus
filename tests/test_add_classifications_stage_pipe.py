#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from morpheus._lib.common import FileTypes
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.messages import MultiResponseMessage
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataframeStage
from morpheus.stages.postprocess.add_classifications_stage import AddClassificationsStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from stages.conv_msg import ConvMsg
from utils import TEST_DIRS


@pytest.mark.slow
def test_add_classifications_stage_pipe(config):
    config.class_labels = ['frogs', 'lizards', 'toads', 'turtles']
    config.num_threads = 1

    threshold = 0.75

    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    input_df = read_file_to_df(input_file, df_type='pandas', file_type=FileTypes.Auto)
    expected_df = (input_df > threshold)

    # Replace input columns with the class labels
    expected_df = expected_df.rename(columns=dict(zip(expected_df.columns, config.class_labels)))

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file, iterative=False))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(ConvMsg(config, input_file))
    pipe.add_stage(AddClassificationsStage(config, threshold=threshold))
    pipe.add_stage(SerializeStage(config, include=["^{}$".format(c) for c in config.class_labels]))
    comp_stage = pipe.add_stage(CompareDataframeStage(config, expected_df))
    pipe.run()

    results = comp_stage.get_results()
    assert results["diff_cols"] == 0, f"Expected diff_cols=0 : {results}"
    assert results["diff_rows"] == 0, f"Expected diff_rows=0 : {results}"


@pytest.mark.slow
def test_add_classifications_stage_multi_segment_pipe(config):
    config.class_labels = ['frogs', 'lizards', 'toads', 'turtles']
    config.num_threads = 1

    threshold = 0.75

    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    input_df = read_file_to_df(input_file, df_type='pandas', file_type=FileTypes.Auto)
    expected_df = (input_df > threshold)

    # Replace input columns with the class labels
    expected_df = expected_df.rename(columns=dict(zip(expected_df.columns, config.class_labels)))

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file, iterative=False))
    pipe.add_segment_boundary(MessageMeta)
    pipe.add_stage(DeserializeStage(config))
    pipe.add_segment_boundary(MultiMessage)
    pipe.add_stage(ConvMsg(config, input_file))
    pipe.add_segment_boundary(MultiResponseMessage)
    pipe.add_stage(AddClassificationsStage(config, threshold=threshold))
    pipe.add_segment_boundary(MultiResponseMessage)
    pipe.add_stage(SerializeStage(config, include=["^{}$".format(c) for c in config.class_labels]))
    pipe.add_segment_boundary(MessageMeta)
    comp_stage = pipe.add_stage(CompareDataframeStage(config, expected_df))
    pipe.run()

    results = comp_stage.get_results()
    assert results["diff_cols"] == 0, f"Expected diff_cols=0 : {results}"
    assert results["diff_rows"] == 0, f"Expected diff_rows=0 : {results}"
