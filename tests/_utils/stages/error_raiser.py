# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import typing

import mrc
from mrc.core import operators as ops

from morpheus.config import Config
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.atomic_integer import AtomicInteger


class ErrorRaiserStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Stage that raises an exception in the on_data method
    """

    def __init__(self, config: Config, exception_cls: type[Exception] = RuntimeError, raise_on: int = 0):
        assert raise_on >= 0

        super().__init__(config)
        self._exception_cls = exception_cls
        self._raise_on = raise_on
        self._counter = AtomicInteger(0)
        self._error_raised = False

    @property
    def name(self) -> str:
        return "error-raiser"

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    def supports_cpp_node(self) -> bool:
        return False

    def on_data(self, message: typing.Any):
        count = self._counter.get_and_inc()
        if count >= self._raise_on:
            self._error_raised = True
            raise self._exception_cls(f"ErrorRaiserStage: raising exception on message {count}")
        return message

    @property
    def error_raised(self) -> bool:
        return self._error_raised

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
