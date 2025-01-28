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
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage


# Adapted from fil_in_out_stage -- used for testing multi-segment error conditions
@pytest.mark.use_cudf
def test_linear_boundary_stages(config, filter_probs_df):
    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipe.add_segment_boundary(MessageMeta)
    comp_stage = pipe.add_stage(CompareDataFrameStage(config, filter_probs_df))
    pipe.run()

    assert_results(comp_stage.get_results())


@pytest.mark.use_cudf
def test_multi_segment_bad_data_type(config, filter_probs_df):
    with pytest.raises(RuntimeError):
        pipe = LinearPipeline(config)
        pipe.set_source(InMemorySourceStage(config, [filter_probs_df]))
        pipe.add_segment_boundary(int)
        mem_sink = pipe.add_stage(InMemorySinkStage(config))
        pipe.run()

    assert len(mem_sink.get_messages()) == 0


def test_add_segment_boundary_as_shared_pointer_error(config, filter_probs_df):
    """
    Test for the assertion error raised when `as_shared_pointer=True` is passed to `add_segment_boundary`.
    Remove this test when the `as_shared_pointer` functionality is implemented.
    """
    with pytest.raises(AssertionError):
        pipe = LinearPipeline(config)
        pipe.set_source(InMemorySourceStage(config, [filter_probs_df]))
        pipe.add_segment_boundary(MessageMeta, as_shared_pointer=True)
