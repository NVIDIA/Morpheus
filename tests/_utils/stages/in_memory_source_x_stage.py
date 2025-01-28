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
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stage_schema import StageSchema


class InMemSourceXStage(GpuAndCpuMixin, SingleOutputSource):
    """
    InMemorySourceStage subclass that emits whatever you give it and doesn't assume the source data
    is a dataframe.
    """

    def __init__(self, c: Config, data: typing.List[typing.Any]):
        super().__init__(c)
        self._data = data

    @property
    def name(self) -> str:
        return "from-data"

    def supports_cpp_node(self) -> bool:
        return False

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(type(self._data[0]))

    def _emit_data(self) -> typing.Iterator[typing.Any]:
        for x in self._data:
            yield x

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:
        return builder.make_source(self.unique_name, self._emit_data())
