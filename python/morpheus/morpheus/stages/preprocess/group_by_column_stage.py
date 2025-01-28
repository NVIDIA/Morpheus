# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

import mrc
from mrc.core import operators as ops

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage


class GroupByColumnStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Group the incoming message by a column in the DataFrame.

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance
    column_name : str
        The column name in the message dataframe to group by
    """

    def __init__(self, config: Config, column_name: str):
        super().__init__(config)

        self._column_name = column_name

    @property
    def name(self) -> str:
        return "group-by-column"

    def accepted_types(self) -> tuple:
        """
        Returns accepted input types for this stage.
        """
        return (MessageMeta, )

    def supports_cpp_node(self) -> bool:
        """
        Indicates whether this stage supports C++ node.
        """
        return False

    def on_data(self, message: MessageMeta) -> list[MessageMeta]:
        """
        Group the incoming message by a column in the DataFrame.

        Parameters
        ----------
        message : MessageMeta
            Incoming message
        """
        with message.mutable_dataframe() as df:
            grouper = df.groupby(self._column_name)

        output_messages = []
        for group_name in sorted(grouper.groups.keys()):
            group_df = grouper.get_group(group_name)
            output_messages.append(MessageMeta(group_df))

        return output_messages

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data), ops.flatten())
        builder.make_edge(input_node, node)

        return node
