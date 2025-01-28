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

import gc
import typing

import pytest

from _utils.stages.control_message_pass_thru import ControlMessagePassThruStage
from _utils.stages.conv_msg import ConvMsg
from _utils.stages.in_memory_multi_source_stage import InMemoryMultiSourceStage
from _utils.stages.in_memory_source_x_stage import InMemSourceXStage
from _utils.stages.multi_port_pass_thru import MultiPortPassThruStage
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline import Pipeline
from morpheus.pipeline.stage_decorator import source
from morpheus.pipeline.stage_decorator import stage
from morpheus.stages.boundary.linear_boundary_stage import LinearBoundaryEgressStage
from morpheus.stages.boundary.linear_boundary_stage import LinearBoundaryIngressStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils.type_aliases import DataFrameType


class SourceTestStage(InMemorySourceStage):  # pylint: disable=too-many-ancestors

    def __init__(self,
                 config,
                 dataframes: typing.List[DataFrameType],
                 on_start_cb: typing.Callable[[], None] = None,
                 start_async_cb: typing.Callable[[], None] = None,
                 destructor_cb: typing.Callable[[], None] = None,
                 repeat: int = 1):
        super().__init__(config, dataframes, repeat)
        self._on_start_cb = on_start_cb
        self._start_async_cb = start_async_cb
        self._destructor_cb = destructor_cb

    @property
    def name(self) -> str:
        return "test-source"

    def on_start(self):
        if self._on_start_cb is not None:
            self._on_start_cb()

    async def start_async(self):
        await super().start_async()
        if self._start_async_cb is not None:
            self._start_async_cb()

    def __del__(self):
        if self._destructor_cb is not None:
            self._destructor_cb()


class SinkTestStage(InMemorySinkStage):

    def __init__(self,
                 config,
                 on_start_cb: typing.Callable[[], None] = None,
                 start_async_cb: typing.Callable[[], None] = None,
                 destructor_cb: typing.Callable[[], None] = None):
        super().__init__(config)
        self._on_start_cb = on_start_cb
        self._start_async_cb = start_async_cb
        self._destructor_cb = destructor_cb

    @property
    def name(self) -> str:
        return "test-sink"

    def on_start(self):
        if self._on_start_cb is not None:
            self._on_start_cb()

    async def start_async(self):
        await super().start_async()
        if self._start_async_cb is not None:
            self._start_async_cb()

    def __del__(self):
        if self._destructor_cb is not None:
            self._destructor_cb()


def _run_pipeline(filter_probs_df: DataFrameType,
                  source_callbacks: dict[str, typing.Callable[[], None]],
                  sink_callbacks: dict[str, typing.Callable[[], None]]):
    config = Config()
    pipe = LinearPipeline(config)
    pipe.set_source(SourceTestStage(config, [filter_probs_df], **source_callbacks))
    pipe.add_stage(SinkTestStage(config, **sink_callbacks))

    with pytest.deprecated_call(match="The on_start method is deprecated and may be removed in the future.*"):
        # The sink stage ensures that the on_start callback method still works, even though it is deprecated.
        pipe.run()


@pytest.mark.use_cudf
def test_destructors_called(filter_probs_df: DataFrameType):
    """
    Test to ensure that the destructors of stages are called (issue #1114).
    """
    state_dict = {"source": False, "sink": False}

    def update_state_dict(key: str):
        nonlocal state_dict
        state_dict[key] = True

    source_callbacks = {'destructor_cb': lambda: update_state_dict("source")}
    sink_callbacks = {'destructor_cb': lambda: update_state_dict("sink")}

    _run_pipeline(filter_probs_df, source_callbacks, sink_callbacks)

    gc.collect()
    assert state_dict["source"]
    assert state_dict["sink"]


@pytest.mark.use_cudf
def test_startup_cb_called(filter_probs_df: DataFrameType):
    """
    Test to ensure that the destructors of stages are called (issue #1114).
    """
    state_dict = {
        "source_on_start": False, "source_start_async": False, "sink_on_start": False, "sink_start_async": False
    }

    def update_state_dict(key: str):
        nonlocal state_dict
        state_dict[key] = True

    source_callbacks = {
        'on_start_cb': lambda: update_state_dict("source_on_start"),
        'start_async_cb': lambda: update_state_dict("source_start_async")
    }

    sink_callbacks = {
        'on_start_cb': lambda: update_state_dict("sink_on_start"),
        'start_async_cb': lambda: update_state_dict("sink_start_async")
    }

    _run_pipeline(filter_probs_df, source_callbacks=source_callbacks, sink_callbacks=sink_callbacks)

    assert state_dict["source_on_start"]
    assert state_dict["source_start_async"]
    assert state_dict["sink_on_start"]
    assert state_dict["sink_start_async"]


@pytest.mark.use_cudf
def test_pipeline_narrowing_types(config: Config):
    """
    Test to ensure that we aren't narrowing the types of messages in the pipeline.
    In this case, `derived_control_message_source` emits `DerivedControlMessage` messages which are a (dummy)
    subclass of `ControlMessage`, which is the accepted type for `ControlMessagePassThruStage`.
    We want to ensure that the type is retained allowing us to place a stage after `ControlMessagePassThruStage`
    requring `DerivedControlMessage`.
    """
    pipe = LinearPipeline(config)

    class DerivedControlMessage(ControlMessage):
        pass

    @source
    def derived_control_message_source() -> DerivedControlMessage:
        yield DerivedControlMessage()

    @stage
    def derived_control_message_sink(msg: DerivedControlMessage) -> DerivedControlMessage:
        return msg

    pipe.set_source(derived_control_message_source(config))  # pylint: disable=E1121
    pipe.add_stage(ControlMessagePassThruStage(config))
    pipe.add_stage(derived_control_message_sink(config))
    pipe.run()


@pytest.mark.parametrize("num_outputs", [0, 2, 3])
def test_add_edge_output_port_errors(config: Config, num_outputs: int):
    """
    Calling add_edge where start has either no output ports or multiple output ports should cause an assertion error.
    """
    data = [list(range(3)) for _ in range(num_outputs)]
    start_stage = InMemoryMultiSourceStage(config, data=data)

    pipe = Pipeline(config)
    pipe.add_stage(start_stage)
    end_stage = pipe.add_stage(ConvMsg(config))

    with pytest.raises(AssertionError):
        pipe.add_edge(start_stage, end_stage.input_ports[0])


@pytest.mark.parametrize("num_inputs", [0, 2, 3])
def test_add_edge_input_port_errors(config: Config, num_inputs: int):
    """
    Calling add_edge where end has either no input ports or multiple input ports should cause an assertion error.
    """
    start_stage = InMemSourceXStage(config, data=list(range(3)))
    end_stage = MultiPortPassThruStage(config, num_ports=num_inputs)

    pipe = Pipeline(config)
    pipe.add_stage(start_stage)
    pipe.add_stage(end_stage)

    with pytest.raises(AssertionError):
        pipe.add_edge(start_stage.output_ports[0], end_stage)


@pytest.mark.parametrize("data_type", [int, float, str, MessageMeta, ControlMessage])
def test_add_segment_edge(config: Config, data_type: type):
    pipe = Pipeline(config)

    boundary_egress = LinearBoundaryEgressStage(config, boundary_port_id="seg_1", data_type=data_type)
    boundary_ingress = LinearBoundaryIngressStage(config, boundary_port_id="seg_1", data_type=data_type)

    pipe.add_stage(boundary_egress, "seg_1")
    pipe.add_stage(boundary_ingress, "seg_2")
    pipe.add_segment_edge(boundary_egress, "seg_1", boundary_ingress, "seg_2", ("seg_1", object, False))


def test_add_segment_edge_assert_not_built(config: Config):
    pipe = Pipeline(config)

    src_stage = InMemSourceXStage(config, data=list(range(3)))
    boundary_egress = LinearBoundaryEgressStage(config, boundary_port_id="seg_1", data_type=int)
    boundary_ingress = LinearBoundaryIngressStage(config, boundary_port_id="seg_1", data_type=int)

    pipe.add_stage(src_stage, "seg_1")
    pipe.add_stage(boundary_egress, "seg_1")
    pipe.add_edge(src_stage, boundary_egress, "seg_1")
    pipe.add_stage(boundary_ingress, "seg_2")
    pipe.build()

    with pytest.raises(AssertionError):
        pipe.add_segment_edge(boundary_egress, "seg_1", boundary_ingress, "seg_2", ("seg_1", object, False))


def test_add_segment_edge_bad_egress(config: Config):
    pipe = Pipeline(config)

    bad_egress = InMemorySinkStage(config)
    boundary_ingress = LinearBoundaryIngressStage(config, boundary_port_id="seg_1", data_type=int)

    pipe.add_stage(bad_egress, "seg_1")
    pipe.add_stage(boundary_ingress, "seg_2")

    with pytest.raises(AssertionError):
        pipe.add_segment_edge(bad_egress, "seg_1", boundary_ingress, "seg_2", ("seg_1", object, False))


def test_add_segment_edge_bad_ingress(config: Config):
    pipe = Pipeline(config)

    boundary_egress = LinearBoundaryEgressStage(config, boundary_port_id="seg_1", data_type=int)
    bad_ingress = InMemSourceXStage(config, data=list(range(3)))

    pipe.add_stage(boundary_egress, "seg_1")
    pipe.add_stage(bad_ingress, "seg_2")

    with pytest.raises(AssertionError):
        pipe.add_segment_edge(boundary_egress, "seg_1", bad_ingress, "seg_2", ("seg_1", object, False))
