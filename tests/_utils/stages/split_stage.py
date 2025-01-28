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

import mrc
import mrc.core.operators as ops
from mrc.core.node import Broadcast

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.stage import Stage
from morpheus.pipeline.stage_schema import StageSchema


class SplitStage(GpuAndCpuMixin, Stage):

    def __init__(self, c: Config):
        super().__init__(c)

        self._create_ports(1, 2)

    @property
    def name(self) -> str:
        return "split"

    def supports_cpp_node(self):
        return False

    def compute_schema(self, schema: StageSchema):
        assert len(schema.output_schemas) == 2, "Expected two output schemas"
        for port_schema in schema.output_schemas:
            port_schema.set_type(MessageMeta)

    def _build(self, builder: mrc.Builder, input_nodes: list[mrc.SegmentObject]) -> list[mrc.SegmentObject]:

        assert len(input_nodes) == 1, "Only 1 input supported"

        # Create a broadcast node
        broadcast = Broadcast(builder, "broadcast")
        builder.make_edge(input_nodes[0], broadcast)

        def filter_higher_fn(data: MessageMeta):
            return MessageMeta(data.df[data.df["v2"] >= 0.5])

        def filter_lower_fn(data: MessageMeta):
            return MessageMeta(data.df[data.df["v2"] < 0.5])

        # Create a node that only passes on rows >= 0.5
        filter_higher = builder.make_node("filter_higher", ops.map(filter_higher_fn))
        builder.make_edge(broadcast, filter_higher)

        # Create a node that only passes on rows < 0.5
        filter_lower = builder.make_node("filter_lower", ops.map(filter_lower_fn))
        builder.make_edge(broadcast, filter_lower)

        return [filter_higher, filter_lower]
