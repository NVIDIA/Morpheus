#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import mrc
import mrc.core.operators as ops

from morpheus.config import Config
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.stage import Stage


class MultiPortPassThruStage(GpuAndCpuMixin, PassThruTypeMixin, Stage):

    def __init__(self, c: Config, num_ports: int):
        super().__init__(c)
        self._create_ports(num_ports, num_ports)
        self.num_ports = num_ports

    @property
    def name(self) -> str:
        return "multi-pass-thru"

    def supports_cpp_node(self):
        return False

    def on_data(self, message: typing.Any):
        # Return the message for the next stage
        return message

    def _build(self, builder: mrc.Builder, input_nodes: list[mrc.SegmentObject]) -> list[mrc.SegmentObject]:
        assert self.num_ports == len(input_nodes)

        output_nodes = []

        for (port_idx, input_node) in enumerate(input_nodes):
            node = builder.make_node(f"{self.unique_name}_port_{port_idx}", ops.map(self.on_data))
            builder.make_edge(input_node, node)
            output_nodes.append(node)

        return output_nodes
