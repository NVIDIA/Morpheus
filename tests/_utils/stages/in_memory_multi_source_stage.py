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

from morpheus.config import Config
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.source_stage import SourceStage
from morpheus.pipeline.stage_schema import StageSchema


class InMemoryMultiSourceStage(GpuAndCpuMixin, SourceStage):
    """
    In memory multi-source stage for testing purposes, accepts a 2d array `data`.
    The first dimenion represents the number of output ports, and the second represents the data for each port, and
    is assumed to be of a consistent type per dimension. For example, it is acceptable for data[0] to be a list of
    ints, and data[1] to be a list of strings.
    """

    def __init__(self, c: Config, data: list[list[typing.Any]]):
        super().__init__(c)
        self._create_ports(0, len(data))
        self._data = data

    @property
    def name(self) -> str:
        return "multi-in-memory-source"

    def supports_cpp_node(self) -> bool:
        return False

    def compute_schema(self, schema: StageSchema):
        assert len(self._data) == len(schema.output_schemas), "Number of output ports must match number of data arrays"
        for (port_idx, port_schema) in enumerate(schema.output_schemas):
            port_schema.set_type(type(self._data[port_idx][0]))

    def _emit_data(self) -> typing.Iterator[typing.Any]:
        for x in self._data:
            yield x

    def _build_sources(self, builder: mrc.Builder) -> list[mrc.SegmentObject]:
        return [builder.make_source(self.unique_name, self._emit_data()) for _ in range(len(self._data))]
