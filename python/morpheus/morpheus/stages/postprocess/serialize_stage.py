# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
from functools import partial

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.controllers.serialize_controller import SerializeController
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema

logger = logging.getLogger(__name__)


@register_stage("serialize")
class SerializeStage(GpuAndCpuMixin, SinglePortStage):
    """
    Includes & excludes columns from messages.

    This class filters columns from a `ControlMessage` object emitting a `MessageMeta`.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    include : typing.List[str], default = [], show_default="All Columns",
        Attributes that are required send to downstream stage.
    exclude : typing.List[str], default = [r'^ID$', r'^_ts_']
        Attributes that are not required send to downstream stage.
    fixed_columns : bool
        When `True` `SerializeStage` will assume that the Dataframe in all messages contain the same columns as the
        first message received.
    """

    def __init__(self,
                 config: Config,
                 include: typing.List[str] = None,
                 exclude: typing.List[str] = None,
                 fixed_columns: bool = True):
        super().__init__(config)

        if (include is None):
            include = []

        if (exclude is None):
            exclude = [r'^ID$', r'^_ts_']

        self._controller = SerializeController(include=include, exclude=exclude, fixed_columns=fixed_columns)

    @property
    def name(self) -> str:
        return "serialize"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple(`morpheus.messages.ControlMessage`)
            Accepted input types.

        """
        return (ControlMessage, )

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(MessageMeta)

    def supports_cpp_node(self):
        # Enable support by default
        return True

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        if (self._build_cpp_node()):
            import morpheus._lib.stages as _stages
            node = _stages.SerializeStage(builder,
                                          self.unique_name,
                                          self._controller.include_columns or [],
                                          self._controller.exclude_columns,
                                          self._controller.fixed_columns)
        else:
            include_columns = self._controller.get_include_col_pattern()
            exclude_columns = self._controller.get_exclude_col_pattern()

            node = builder.make_node(
                self.unique_name,
                ops.map(
                    partial(self._controller.convert_to_df,
                            include_columns=include_columns,
                            exclude_columns=exclude_columns)))

        builder.make_edge(input_node, node)

        return node
