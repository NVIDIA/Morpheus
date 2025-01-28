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

import pytest

from _utils import assert_results
from _utils.stages.conv_msg import ConvMsg
from morpheus.common import FilterSource
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.postprocess.filter_detections_stage import FilterDetectionsStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage


@pytest.mark.slow
@pytest.mark.use_cudf
@pytest.mark.parametrize('use_conv_msg', [True, False])
@pytest.mark.parametrize('do_copy', [True, False])
@pytest.mark.parametrize('threshold', [0.1, 0.5, 0.8])
@pytest.mark.parametrize('field_name', ['v1', 'v2', 'v3', 'v4'])
def test_filter_column(config, filter_probs_df, use_conv_msg, do_copy, threshold, field_name):
    expected_df = filter_probs_df.to_pandas()
    expected_df = expected_df[expected_df[field_name] > threshold]

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipe.add_stage(DeserializeStage(config))

    # When `use_conv_msg` is true, ConvMsg will convert messages to MultiResponseProbs,
    # when false, the filter stage will receive instances of ControlMessage
    if use_conv_msg:
        pipe.add_stage(ConvMsg(config, empty_probs=True))

    pipe.add_stage(
        FilterDetectionsStage(config,
                              threshold=threshold,
                              copy=do_copy,
                              filter_source=FilterSource.DATAFRAME,
                              field_name=field_name))
    pipe.add_stage(SerializeStage(config))
    comp_stage = pipe.add_stage(CompareDataFrameStage(config, expected_df))
    pipe.run()

    assert_results(comp_stage.get_results())
