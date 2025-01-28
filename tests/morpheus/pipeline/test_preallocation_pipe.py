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

import numpy as np
import pandas as pd
import pytest

from _utils import assert_results
from _utils.stages.check_pre_alloc import CheckPreAlloc
from _utils.stages.conv_msg import ConvMsg
from morpheus.common import TypeId
from morpheus.common import typeid_to_numpy_str
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.postprocess.add_scores_stage import AddScoresStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.type_aliases import DataFrameType


@pytest.mark.gpu_and_cpu_mode
@pytest.mark.parametrize('probs_type', [TypeId.FLOAT32, TypeId.FLOAT64])
def test_preallocation(config: Config, filter_probs_df: DataFrameType, probs_type: TypeId):
    config.class_labels = ['frogs', 'lizards', 'toads', 'turtles']
    probs_np_type = typeid_to_numpy_str(probs_type)
    expected_df = pd.DataFrame(
        data={c: np.zeros(len(filter_probs_df), dtype=probs_np_type)
              for c in config.class_labels})

    pipe = LinearPipeline(config)
    mem_src = pipe.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(ConvMsg(config, columns=list(filter_probs_df.columns), probs_type=probs_np_type))
    pipe.add_stage(CheckPreAlloc(config, probs_type=probs_type))
    pipe.add_stage(SerializeStage(config, include=[f"^{c}$" for c in config.class_labels]))
    comp_stage = pipe.add_stage(CompareDataFrameStage(config, expected_df))

    assert len(mem_src.get_needed_columns()) == 0

    pipe.run()

    assert mem_src.get_needed_columns() == {
        'frogs': probs_type, 'lizards': probs_type, 'toads': probs_type, 'turtles': probs_type
    }

    assert_results(comp_stage.get_results())


@pytest.mark.gpu_and_cpu_mode
@pytest.mark.parametrize('probs_type', [TypeId.FLOAT32, TypeId.FLOAT64])
def test_preallocation_multi_segment_pipe(config: Config, filter_probs_df: DataFrameType, probs_type: TypeId):
    """
    Test ensures that when columns are needed for preallocation in a multi-segment pipeline, the preallocagtion will
    always be performed on the closest source to the stage that requested preallocation. Which in cases where the
    requesting stage is not in the first segment, then the preallocation will be performed on the segment ingress
    """
    config.class_labels = ['frogs', 'lizards', 'toads', 'turtles']
    probs_np_type = typeid_to_numpy_str(probs_type)
    expected_df = pd.DataFrame(
        data={c: np.zeros(len(filter_probs_df), dtype=probs_np_type)
              for c in config.class_labels})

    pipe = LinearPipeline(config)
    mem_src = pipe.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipe.add_segment_boundary(MessageMeta)
    pipe.add_stage(DeserializeStage(config))
    pipe.add_segment_boundary(ControlMessage)
    pipe.add_stage(ConvMsg(config, columns=list(filter_probs_df.columns), probs_type=typeid_to_numpy_str(probs_type)))
    (_, boundary_ingress) = pipe.add_segment_boundary(ControlMessage)
    pipe.add_stage(CheckPreAlloc(config, probs_type=probs_type))
    pipe.add_segment_boundary(ControlMessage)
    pipe.add_stage(SerializeStage(config, include=[f"^{c}$" for c in config.class_labels]))
    pipe.add_segment_boundary(MessageMeta)
    comp_stage = pipe.add_stage(CompareDataFrameStage(config, expected_df))

    assert len(mem_src.get_needed_columns()) == 0

    pipe.run()

    assert len(mem_src.get_needed_columns()) == 0
    assert boundary_ingress.get_needed_columns() == {
        'frogs': probs_type, 'lizards': probs_type, 'toads': probs_type, 'turtles': probs_type
    }

    assert_results(comp_stage.get_results())


@pytest.mark.gpu_mode
def test_preallocation_error(config, filter_probs_df):
    """
    Verify that we get a raised exception when add_scores attempts to use columns that don't exist
    """
    config.class_labels = ['frogs', 'lizards', 'toads', 'turtles']

    pipe = LinearPipeline(config)
    mem_src = pipe.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipe.add_stage(DeserializeStage(config, ensure_sliceable_index=True))
    pipe.add_stage(ConvMsg(config, columns=list(filter_probs_df.columns), probs_type='f4'))
    add_scores = pipe.add_stage(AddScoresStage(config))
    pipe.add_stage(SerializeStage(config, include=[f"^{c}$" for c in config.class_labels]))
    mem_sink = pipe.add_stage(InMemorySinkStage(config))

    assert len(mem_src.get_needed_columns()) == 0
    assert len(add_scores.get_needed_columns()) > 0
    add_scores._needed_columns = {}
    assert len(add_scores.get_needed_columns()) == 0

    try:
        pipe.run()
    except Exception as e:
        assert isinstance(e, RuntimeError)

        # Ensure the error mentioned populating the needed_columns and using a stage with the PreallocatorMixin
        # Without depending on a specific string
        assert "frogs" in str(e)
        assert "PreallocatorMixin" in str(e)
        assert "needed_columns" in str(e)

    assert len(mem_src.get_needed_columns()) == 0
    assert len(mem_sink.get_messages()) == 0
