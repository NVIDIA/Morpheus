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

from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_ids import TO_CONTROL_MESSAGE
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)


@register_module(TO_CONTROL_MESSAGE, MORPHEUS_MODULE_NAMESPACE)
def to_control_message(builder: mrc.Builder):
    """
    This module converts MessageMeta to a ControlMessage.

    Parameters
    ----------
    builder : mrc.Builder
        An mrc Builder object.

    Notes
    -----
        Configurable Parameters:
            - control_message_conf (dict): Control message configuration;
                Example: `{"meta_data": {"data_type": "streaming"}, "tasks": [{"type": "inference", "properties": {}}]}`
                Default: `{}`
    """

    config = builder.get_current_module_config()

    control_message_conf = config.get("control_message_conf", {})

    if not isinstance(control_message_conf, dict):
        raise TypeError(
            f"Expected dictionary type for 'control_message_conf' but recieved: {type(control_message_conf)}")

    def meta_to_control_message(x: MessageMeta) -> ControlMessage:
        # Set control message configuration
        control_message = ControlMessage(control_message_conf)
        control_message.payload(x)

        return control_message

    node = builder.make_node(TO_CONTROL_MESSAGE, ops.map(meta_to_control_message))

    # Register input and output port for a module.
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
