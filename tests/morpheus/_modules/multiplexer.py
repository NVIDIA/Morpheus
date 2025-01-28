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

import mrc
from mrc.core import operators as ops

from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)


@register_module("multiplexer", "morpheus_test")
def multiplexer(builder: mrc.Builder):
    """
    The multiplexer receives data packets from one or more input ports and interleaves them into a single output.

    Parameters
    ----------
    builder : mrc.Builder
        mrc Builder object.

    Notes
    -----
        Configurable Parameters:
            - input_ports (list(str)): Input ports data streams to be combined; Example: `["intput_1", "input_2"]`;
            Default: None
            - output_port (str): Output port where the combined streams to be passed; Example: `"output"`;
            Default: None
    """

    config = builder.get_current_module_config()

    input_ports = config.get("input_ports", None)
    output_port = config.get("output_port", None)

    if not input_ports:
        raise ValueError("The 'input_ports' parameter must be set with at least one value.")
    if not output_port:
        raise ValueError("The 'output_port' parameter must be set.")

    def on_next(data):
        return data

    # Make an output node.
    output_node = builder.make_node(output_port, ops.map(on_next))

    # Loop through the provided input ports, creating a node for each one and directing the data streams
    # of all input ports to a single output node.
    for input_port in input_ports:
        # Make an input node.
        input_node = builder.make_node(input_port, ops.map(on_next))
        # Make an edge betweeen input node and an output node.
        builder.make_edge(input_node, output_node)
        # Register an input port for a module.
        builder.register_module_input(input_port, input_node)

    # Register an output port for a module.
    builder.register_module_output(output_port, output_node)
