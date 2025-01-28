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

import typing

import mrc
import mrc.core.operators as ops

from morpheus.config import Config
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage


class InMemorySinkStage(PassThruTypeMixin, GpuAndCpuMixin, SinglePortStage):
    """
    Collects incoming messages into a list that can be accessed after the pipeline is complete. Useful for testing.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._messages = []

    @property
    def name(self) -> str:
        return "to-mem"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple(`typing.Any`, )
            Accepted input types.

        """
        return (typing.Any, )

    def supports_cpp_node(self) -> bool:
        return False

    def clear(self):
        """
        Clear the messages that have been collected so far
        """
        self._messages.clear()

    def get_messages(self) -> typing.List[typing.Any]:
        """
        Returns
        -------
        list
            Results collected messages
        """
        return self._messages

    def _append_message(self, message: typing.Any):
        self._messages.append(message)
        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self._append_message))
        builder.make_edge(input_node, node)

        return node
