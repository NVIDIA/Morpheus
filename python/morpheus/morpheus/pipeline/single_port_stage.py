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
from abc import abstractmethod

import mrc
import typing_utils

import morpheus.pipeline as _pipeline  # pylint: disable=cyclic-import
from morpheus.config import Config
from morpheus.utils.type_utils import pretty_print_type_name

logger = logging.getLogger(__name__)


class SinglePortStage(_pipeline.Stage):
    """
    Class used for building stages with single input port and single output port.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._create_ports(1, 1)

    @abstractmethod
    def accepted_types(self) -> tuple:
        """
        Accepted input types for this stage are returned. Derived classes should override this method. An
        error will be generated if the input types to the stage do not match one of the available types
        returned from this method.

        Returns
        -------
        tuple
            Accepted input types.

        """
        pass

    def _pre_compute_schema(self, schema: _pipeline.StageSchema):
        # Pre-flight check to verify that the input type is one of the accepted types
        super()._pre_compute_schema(schema)
        accepted_types = typing.Union[self.accepted_types()]
        input_type = schema.input_type
        if (not typing_utils.issubtype(input_type, accepted_types)):
            raise RuntimeError((f"The {self.name} stage cannot handle input of {input_type}. "
                                f"Accepted input types: {self.accepted_types()}"))

    @abstractmethod
    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        pass

    def _build(self, builder: mrc.Builder, input_nodes: list[mrc.SegmentObject]) -> list[mrc.SegmentObject]:
        # Derived source stages should override `_build_single` instead of this method. This allows for tracking the
        # True source object separate from the output node. If any other operators need to be added after the node,
        # use `_post_build`
        assert len(self.input_ports) == 1 and len(self.output_ports) == 1, \
            "SinglePortStage must have 1 input port and 1 output port"

        assert len(input_nodes) == 1, "Should only have 1 input node"

        return [self._build_single(builder, input_nodes[0])]

    def _post_build_single(self, _: mrc.Builder, out_node: mrc.SegmentObject) -> mrc.SegmentObject:
        return out_node

    def _post_build(self, builder: mrc.Builder, out_ports_nodes: list[mrc.SegmentObject]) -> list[mrc.SegmentObject]:

        ret_val = self._post_build_single(builder, out_ports_nodes[0])

        # pylint: disable=logging-format-interpolation
        logger.info("Added stage: %s\n  └─ %s -> %s",
                    str(self),
                    pretty_print_type_name(self.input_ports[0].input_type),
                    pretty_print_type_name(self.output_ports[0].output_type))

        return [ret_val]
