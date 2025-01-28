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
"""This module converts ControlMessage to a MessageMeta."""
import logging

import mrc
from mrc.core import operators as ops

from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.module_ids import FROM_CONTROL_MESSAGE
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)


@register_module(FROM_CONTROL_MESSAGE, MORPHEUS_MODULE_NAMESPACE)
def from_control_message(builder: mrc.Builder):
    """
    This module converts ControlMessage to a MessageMeta.

    Parameters
    ----------
    builder : mrc.Builder
        An mrc Builder object.
    """

    def control_message_to_meta(x: ControlMessage) -> MessageMeta:
        if not isinstance(x, ControlMessage):
            raise TypeError(f"Expected 'x' to be of type ControlMessage, but instead got {type(x).__name__}.")

        message_meta = x.payload()
        if message_meta is None:
            logger.debug("ControlMessage does not contain a payload, it cannot be converted to a MessageMeta object."
                         " Skipping conversion process")

        return message_meta

    node = builder.make_node(FROM_CONTROL_MESSAGE,
                             ops.map(control_message_to_meta),
                             ops.filter(lambda x: x is not None))

    # Register input and output port for a module.
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
