# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

import logging
import typing

import mrc

from morpheus.config import Config
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stage_schema import StageSchema

logger = logging.getLogger(f"morpheus.{__name__}")

DataSourceType = (typing.Callable[[mrc.Subscription], typing.Iterable[typing.Any]]
                  | typing.Callable[[], typing.Iterable[typing.Any]])


class InMemoryDataGenStage(GpuAndCpuMixin, SingleOutputSource):
    """
    Source stage that generates data in-memory using a provided iterable or generator function.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    data_source : Callable[[mrc.Subscription], Iterable[Any]]
        An iterable or a generator function that yields data to be processed by the pipeline.
    output_data_type : Type
        The data type of the objects that the data_source yields.
    """

    def __init__(self, c: Config, data_source: DataSourceType, output_data_type: typing.Type = typing.Any):
        super().__init__(c)
        self._data_source = data_source
        self._output_data_type = output_data_type

    @property
    def name(self) -> str:
        return "in-memory-data-gen"

    def compute_schema(self, schema: StageSchema):
        # Set the output schema based on the OutputDataType
        schema.output_schema.set_type(self._output_data_type)

    def supports_cpp_node(self) -> bool:
        return False

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:
        return builder.make_source(self.unique_name, self._data_source)
