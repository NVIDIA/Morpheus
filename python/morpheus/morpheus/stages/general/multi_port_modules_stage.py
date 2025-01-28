# Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
"""MultiPortModulesStage class."""

import logging
import typing

import mrc

from morpheus.config import Config
from morpheus.pipeline.stage import Stage
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.utils.module_utils import load_module

logger = logging.getLogger(__name__)


class MultiPortModulesStage(Stage):
    """
    Loads an existing, registered, MRC SegmentModule and wraps it as a Morpheus Stage.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    module_config : typing.Dict
        Module configuration.
    input_ports : typing.List[str]
        Input ports used for the registered module.
    output_ports : typing.List[str]
        Output ports used for the registered module.
    input_type : default `typing.Any`
        The stage acceptable input type.
    output_type : default `typing.Any`
        The output type that the stage produces.

    """

    def __init__(self,
                 c: Config,
                 module_conf: typing.Dict[str, typing.Any],
                 input_ports: typing.List[str],
                 output_ports: typing.List[str],
                 input_type=typing.Any,
                 output_type=typing.Any):

        super().__init__(c)

        self._module_conf = module_conf
        self._input_type = input_type
        self._ouput_type = output_type

        if not (input_ports and output_ports):
            raise ValueError("Module input and output ports must not be empty.")
        self._in_ports = input_ports
        self._out_ports = output_ports

        self._num_in_ports = len(self._in_ports)
        self._num_out_ports = len(self._out_ports)

        self._create_ports(self._num_in_ports, self._num_out_ports)

    @property
    def name(self) -> str:
        """Returns the name of the stage."""
        return self._module_conf.get("module_name", "multi_port_module")

    def supports_cpp_node(self):
        """Indicates whether the stage supports C++ node."""
        return False

    def input_types(self) -> typing.Tuple:
        """
        Returns input type for the current stage.
        """

        return (self._input_type, )

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple
            Accepted input types.

        """
        return (typing.Any, )

    def compute_schema(self, schema: StageSchema):
        for port_schema in schema.output_schemas:
            port_schema.set_type(self._ouput_type)

    def _validate_ports(self, module) -> None:

        input_ids = module.input_ids()
        output_ids = module.output_ids()

        if sorted(self._in_ports) != sorted(input_ids):
            raise ValueError(f"Provided input ports do not match module input ports. Module: {input_ids}, "
                             f"Provided: {self._in_ports}.")

        if sorted(self._out_ports) != sorted(output_ids):
            raise ValueError(f"Provided output ports do not match module output ports. Module: {output_ids}, "
                             f"Provided: {self._out_ports}.")

    def _build(self, builder: mrc.Builder, input_nodes: list[mrc.SegmentObject]) -> list[mrc.SegmentObject]:

        # Load module from the registry.
        module = load_module(self._module_conf, builder=builder)

        self._validate_ports(module)

        # Make an edges with input ports
        for index in range(self._num_in_ports):
            in_stream_node = input_nodes[index]
            in_port = self._in_ports[index]
            mod_in_stream = module.input_port(in_port)
            builder.make_edge(in_stream_node, mod_in_stream)

        out_nodes = []

        for index in range(self._num_out_ports):
            out_port = self._out_ports[index]
            out_nodes.append(module.output_port(out_port))

        return out_nodes
