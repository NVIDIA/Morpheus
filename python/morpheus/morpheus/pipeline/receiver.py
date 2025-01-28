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
from morpheus.utils.type_utils import greatest_ancestor

logger = logging.getLogger(__name__)


class Receiver():
    """
    The `Receiver` object represents a downstream port on a `StageBase` object that gets messages from a `Sender`.

    Parameters
        ----------
        parent : `morpheus.pipeline.pipeline.StageBase`
            Parent `StageBase` object.
        port_number : int
            Receiver port number.
    """

    def __init__(self, parent: "_pipeline.StageBase", port_number: int):

        self._parent = parent
        self.port_number = port_number

        self._is_schema_linked = False
        self._is_node_linked = False

        self._input_schema: _pipeline.PortSchema = None
        self._input_node: mrc.SegmentObject = None

        self._input_senders: typing.List[_pipeline.Sender] = []

    @property
    def parent(self):
        return self._parent

    @property
    def is_complete(self):
        """
        A receiver is complete if all input senders are also complete.
        """
        return all(x.is_complete for x in self._input_senders)

    @property
    def is_partial(self):
        """
        A receiver is partially complete if any input sender is complete. Receivers are usually partially complete if
        there is a circular pipeline.
        """
        # Its partially complete if any input sender is complete
        return any(x.is_complete for x in self._input_senders)

    @property
    def input_schema(self) -> _pipeline.PortSchema:
        return self._input_schema

    def get_input_node(self, builder: mrc.Builder) -> mrc.SegmentObject:
        """
        Returns the input or parent node.
        """

        assert self.is_partial, "Must be partially complete to get the input node!"

        # Build the input from the senders
        if (self._input_node is None):
            # First check if we only have 1 input sender
            if (len(self._input_senders) == 1):
                # In this case, our input type is determined from the sole Sender
                sender = self._input_senders[0]

                if sender.output_node is not None:
                    self._input_node = sender.output_node
                    self._is_node_linked = True
            else:
                # We have multiple senders. Create a dummy node to connect all senders
                self._input_node = builder.make_node_component(
                    f"{self.parent.unique_name}-reciever[{self.port_number}]", mrc.core.operators.map(lambda x: x))

                if (self.is_complete):
                    # Connect all streams now
                    for input_sender in self._input_senders:
                        builder.make_edge(input_sender.output_node, self._input_node)

                    self._is_node_linked = True

        return self._input_node

    def _compute_input_schema(self):
        great_ancestor = greatest_ancestor(*[x.output_schema.get_type() for x in self._input_senders if x.is_complete])

        if (great_ancestor is None):
            raise RuntimeError((f"Cannot determine single type for senders of input port for {self._parent}. "
                                "Use a merge stage to handle different types of inputs."))

        self._input_schema = _pipeline.PortSchema(port_type=great_ancestor)
        self._input_schema._complete()
        self._is_schema_linked = True

    def get_input_schema(self) -> _pipeline.PortSchema:
        assert self.is_partial, "Must be partially complete to get the input type!"

        # Build the input from the senders
        if (self._input_schema is None):
            # First check if we only have 1 input sender
            if (len(self._input_senders) == 1):
                # In this case, our input type is determined from the sole Sender
                sender = self._input_senders[0]
                self._input_schema = sender.output_schema
                self._is_schema_linked = True
                if sender.output_node is not None:
                    self._input_node = sender.output_node
                    self._is_node_linked = True
            else:
                # Now determine the output type from what we have
                self._compute_input_schema()

        return self._input_schema

    @property
    def input_type(self) -> type:
        """
        Returns the the upstream node's output type, and in case of multiple upstreams this will return the common
        ancestor type.
        """
        return self.get_input_schema().get_type()

    def link_schema(self):
        """
        The type linking phase determines the final type of the `Receiver`.

        Raises:
            RuntimeError: Throws a `RuntimeError` if the predicted input port type determined during the build phase is
            different than the current port type.
        """

        assert self.is_complete, "Must be complete before linking!"

        if (self._is_schema_linked):
            return

        self._compute_input_schema()

    def link_node(self, builder: mrc.Builder):
        """
        The node linking phase connects all underlying stages.
        """

        assert self.is_complete, "Must be complete before linking!"

        if (self._is_node_linked):
            return

        for sender in self._input_senders:
            assert sender.output_node is not self._input_node
            builder.make_edge(sender.output_node, self._input_node)

        self._is_node_linked = True
