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

import typing

import pandas as pd
import pytest

import cudf

from _utils import assert_results
from _utils.dataset_manager import DatasetManager
from _utils.stages.conv_msg import ConvMsg
from morpheus.common import FilterSource
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.postprocess.filter_detections_stage import FilterDetectionsStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage


def build_expected(df: pd.DataFrame, threshold: float):
    """
    Takes a copy of `df` and apply the threshold
    """
    return df[df.max(axis=1) >= threshold]


def _test_filter_detections_stage_pipe(config: Config,
                                       dataset_pandas: DatasetManager,
                                       copy: bool = True,
                                       order: typing.Literal['F', 'C'] = 'K',
                                       pipeline_batch_size: int = 256,
                                       repeat: int = 1):
    config.pipeline_batch_size = pipeline_batch_size

    input_df = dataset_pandas["filter_probs.csv"]
    if repeat > 1:
        input_df = dataset_pandas.repeat(input_df, repeat_count=repeat)

    threshold = 0.75

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [cudf.DataFrame(input_df)]))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(ConvMsg(config, order=order, columns=list(input_df.columns)))
    pipe.add_stage(FilterDetectionsStage(config, threshold=threshold, copy=copy, filter_source=FilterSource.TENSOR))
    pipe.add_stage(SerializeStage(config))
    comp_stage = pipe.add_stage(
        CompareDataFrameStage(config, build_expected(dataset_pandas["filter_probs.csv"], threshold)))
    pipe.run()

    assert_results(comp_stage.get_results())


def _test_filter_detections_control_message_stage_multi_segment_pipe(config: Config,
                                                                     dataset: DatasetManager,
                                                                     copy: bool = True):
    threshold = 0.75

    input_df = dataset["filter_probs.csv"]
    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [input_df]))
    pipe.add_segment_boundary(MessageMeta)
    pipe.add_stage(DeserializeStage(config))
    pipe.add_segment_boundary(data_type=ControlMessage)
    pipe.add_stage(ConvMsg(config))
    pipe.add_segment_boundary(ControlMessage)
    pipe.add_stage(FilterDetectionsStage(config, threshold=threshold, copy=copy, filter_source=FilterSource.TENSOR))
    pipe.add_segment_boundary(ControlMessage)
    pipe.add_stage(SerializeStage(config))
    pipe.add_segment_boundary(MessageMeta)
    comp_stage = pipe.add_stage(CompareDataFrameStage(config, build_expected(dataset["filter_probs.csv"], threshold)))
    pipe.run()

    assert_results(comp_stage.get_results())


@pytest.mark.slow
@pytest.mark.parametrize('order', ['F', 'C'])
@pytest.mark.parametrize('pipeline_batch_size', [256, 1024, 2048])
@pytest.mark.parametrize('repeat', [1, 10, 100])
@pytest.mark.parametrize('do_copy', [True, False])
def test_filter_detections_stage_pipe(config: Config,
                                      dataset_pandas: DatasetManager,
                                      order: typing.Literal['F', 'C'],
                                      pipeline_batch_size: int,
                                      repeat: int,
                                      do_copy: bool):
    return _test_filter_detections_stage_pipe(config, dataset_pandas, do_copy, order, pipeline_batch_size, repeat)


@pytest.mark.slow
@pytest.mark.parametrize('do_copy', [True, False])
def test_filter_detections_control_message_stage_multi_segment_pipe(config: Config,
                                                                    dataset_pandas: DatasetManager,
                                                                    do_copy: bool):
    return _test_filter_detections_control_message_stage_multi_segment_pipe(config, dataset_pandas, do_copy)
