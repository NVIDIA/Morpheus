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
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import TensorMemory
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.utils.type_aliases import DataFrameType
from morpheus.utils.type_utils import get_array_pkg
from morpheus.utils.type_utils import get_df_pkg
from morpheus.utils.type_utils import get_df_pkg_from_obj


@register_stage("unittest-conv-msg", ignore_args=["expected_data"])
class ConvMsg(GpuAndCpuMixin, SinglePortStage):
    """
    Simple test stage to convert a ControlMessage to a ControlMessage with probs tensor.
    Basically a cheap replacement for running an inference stage.

    Setting `expected_data` to a DataFrame will cause the probs array to by populated by the values in the DataFrame.
    Setting `expected_data` to `None` causes the probs array to be a copy of the incoming dataframe.
    Setting `columns` restricts the columns copied into probs to just the ones specified.
    Setting `order` specifies probs to be in either column or row major
    Setting `empty_probs` will create an empty probs array with 3 columns, and the same number of rows as the dataframe
    """

    def __init__(self,
                 c: Config,
                 expected_data: DataFrameType = None,
                 columns: typing.List[str] = None,
                 order: str = 'K',
                 probs_type: str = 'f4',
                 empty_probs: bool = False):
        super().__init__(c)

        self._df_pkg = get_df_pkg(c.execution_mode)
        self._array_pkg = get_array_pkg(c.execution_mode)

        if expected_data is not None:
            assert isinstance(expected_data, self._df_pkg.DataFrame)

        self._expected_data: DataFrameType | None = expected_data
        self._columns = columns
        self._order = order
        self._probs_type = probs_type
        self._empty_probs = empty_probs

    @property
    def name(self) -> str:
        return "test"

    def accepted_types(self) -> typing.Tuple:
        return (ControlMessage, )

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(ControlMessage)

    def supports_cpp_node(self) -> bool:
        return False

    def _conv_message(self, message: ControlMessage) -> ControlMessage:
        if self._expected_data is not None:
            df_pkg = get_df_pkg_from_obj(self._expected_data)
            if (isinstance(self._expected_data, self._df_pkg.DataFrame)):
                df = self._expected_data.copy(deep=True)
            else:
                df = df_pkg.DataFrame(self._expected_data)

        else:
            df: DataFrameType = message.payload().get_data(self._columns)  # type: ignore

        if self._empty_probs:
            probs = self._array_pkg.zeros([len(df), 3], 'float')
        else:
            probs = self._array_pkg.array(df.values, dtype=self._probs_type, copy=True, order=self._order)

        message.tensors(TensorMemory(count=len(probs), tensors={'probs': probs}))
        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self._conv_message))
        builder.make_edge(input_node, node)

        return node
