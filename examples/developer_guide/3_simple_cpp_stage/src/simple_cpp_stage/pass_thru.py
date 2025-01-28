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

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema


@register_stage("pass-thru")
class PassThruStage(PassThruTypeMixin, GpuAndCpuMixin, SinglePortStage):

    def __init__(self, config: Config):
        super().__init__(config)
        self._input_type = None

    @property
    def name(self) -> str:
        return "pass-thru"

    def accepted_types(self) -> tuple:
        return (typing.Any, )

    def supports_cpp_node(self) -> bool:
        return True

    def compute_schema(self, schema: StageSchema):
        super().compute_schema(schema)  # Call PassThruTypeMixin's compute_schema method
        self._input_type = schema.input_type

    def on_data(self, message: typing.Any):
        # Return the message for the next stage
        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        if self._build_cpp_node() and isinstance(self._input_type, ControlMessage):
            from ._lib import pass_thru_cpp

            node = pass_thru_cpp.PassThruStage(builder, self.unique_name)
        else:
            node = builder.make_node(self.unique_name, ops.map(self.on_data))

        builder.make_edge(input_node, node)
        return node
