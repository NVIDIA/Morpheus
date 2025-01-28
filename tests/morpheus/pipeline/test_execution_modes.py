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

import collections.abc
import typing

import mrc
import pytest
from mrc.core import operators as ops

from _utils.stages.conv_msg import ConvMsg
from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.pipeline.execution_mode_mixins import CpuOnlyMixin
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_decorator import source
from morpheus.pipeline.stage_decorator import stage


@source
def gpu_only_source() -> collections.abc.Iterator[int]:
    for i in range(10):
        yield i


@source(execution_modes=(ExecutionMode.CPU, ))
def cpu_only_source() -> collections.abc.Iterator[int]:
    for i in range(10):
        yield i


@source(execution_modes=(ExecutionMode.CPU, ExecutionMode.GPU))
def gpu_cpu_source() -> collections.abc.Iterator[int]:
    for i in range(10):
        yield i


@stage
def gpu_only_stage(message: typing.Any) -> typing.Any:
    return message


@stage(execution_modes=(ExecutionMode.CPU, ))
def cpu_only_stage(message: typing.Any) -> typing.Any:
    return message


@stage(execution_modes=(ExecutionMode.CPU, ExecutionMode.GPU))
def gpu_cpu_stage(message: typing.Any) -> typing.Any:
    return message


class BaseStage(PassThruTypeMixin, SinglePortStage):

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    def supports_cpp_node(self) -> bool:
        return False

    def on_data(self, data: typing.Any) -> typing.Any:
        return data

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node


class CpuOnlyStage(CpuOnlyMixin, BaseStage):

    @property
    def name(self) -> str:
        return "test-cpu-only-stage"


class GpuOnlyStage(BaseStage):

    @property
    def name(self) -> str:
        return "test-gpu-only-stage"


class GpuAndCpuStage(GpuAndCpuMixin, BaseStage):

    @property
    def name(self) -> str:
        return "test-gpu-and-cpu-stage"


@pytest.mark.parametrize("stage_cls, expected_modes",
                         [
                             (GpuOnlyStage, {ExecutionMode.GPU}),
                             (CpuOnlyStage, {ExecutionMode.CPU}),
                             (GpuAndCpuStage, {ExecutionMode.GPU, ExecutionMode.CPU}),
                             (gpu_only_source, {ExecutionMode.GPU}),
                             (cpu_only_source, {ExecutionMode.CPU}),
                             (gpu_cpu_source, {ExecutionMode.GPU, ExecutionMode.CPU}),
                             (gpu_only_stage, {ExecutionMode.GPU}),
                             (cpu_only_stage, {ExecutionMode.CPU}),
                             (gpu_cpu_stage, {ExecutionMode.GPU, ExecutionMode.CPU}),
                         ])
def test_execution_mode_mixins(stage_cls: type[ConvMsg], expected_modes: set):
    # intentionally not using the config fixture so that we can set the execution mode manually
    config = Config()
    if ExecutionMode.CPU in expected_modes:
        config.execution_mode = ExecutionMode.CPU
    else:
        config.execution_mode = ExecutionMode.GPU

    stage_ = stage_cls(config)
    assert set(stage_.supported_execution_modes()) == expected_modes


@pytest.mark.parametrize("stage_cls, execution_mode",
                         [
                             (GpuOnlyStage, ExecutionMode.CPU),
                             (gpu_only_source, ExecutionMode.CPU),
                             (gpu_only_stage, ExecutionMode.CPU),
                             (CpuOnlyStage, ExecutionMode.GPU),
                             (cpu_only_source, ExecutionMode.GPU),
                             (cpu_only_stage, ExecutionMode.GPU),
                         ])
def test_unsupported_mode_error(stage_cls: type[ConvMsg], execution_mode: ExecutionMode):
    # intentionally not using the config fixture so that we can set the execution mode and avoid iterating over
    # python/C++ execution modes
    config = Config()
    config.execution_mode = execution_mode

    with pytest.raises(RuntimeError, match="Unsupported execution mode"):
        stage_ = stage_cls(config)
        stage_._pre_build(do_propagate=False)
