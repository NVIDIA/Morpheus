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

import mrc

import morpheus.pipeline as _pipeline  # pylint: disable=cyclic-import

logger = logging.getLogger(__name__)


class Sender():
    """
    The `Sender` object represents a port on a `StageBase` object that sends messages to a `Receiver`.

    Parameters
        ----------
        parent : `morpheus.pipeline.pipeline.StageBase`
            Parent `StageBase` object.
        port_number : int
            Sender port number.
    """

    def __init__(self, parent: "_pipeline.StageBase", port_number: int):

        self._parent = parent
        self.port_number = port_number

        self._output_receivers: typing.List[_pipeline.Receiver] = []

        self._output_schema: _pipeline.PortSchema = None
        self._output_node: mrc.SegmentObject = None

    @property
    def parent(self) -> "_pipeline.StageBase":
        return self._parent

    @property
    def is_complete(self) -> bool:
        # Sender is complete when the type has been set
        return self._output_schema is not None

    @property
    def output_schema(self) -> _pipeline.PortSchema:
        return self._output_schema

    @output_schema.setter
    def output_schema(self, value: _pipeline.PortSchema):
        self._output_schema = value

    @property
    def output_type(self) -> type:
        if self.is_complete:
            return self._output_schema.get_type()

        return None

    @property
    def output_node(self) -> mrc.SegmentObject:
        return self._output_node
