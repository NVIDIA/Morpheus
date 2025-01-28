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
"""Base class for single output source stages which support both `MessageMeta` and `ControlMessage` as output types."""
from enum import Enum

import morpheus.pipeline as _pipeline  # pylint: disable=cyclic-import
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta


class SupportedMessageTypes(Enum):
    """Supported output message types"""
    MESSAGE_META = "MessageMeta"
    CONTROL_MESSAGE = "ControlMessage"


class ConfigurableOutputSource(_pipeline.SingleOutputSource):
    """
    Base class for single output source stages which support both `MessageMeta` and `ControlMessage` as output types.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration instance.
    message_type : `SupportedMessageTypes`
        The type of message to emit.
    task_type : str, default = None
        If specified, adds the specified task to the `ControlMessage`. This parameter is only valid when `message_type`
        is set to `CONTROL_MESSAGE`. If not `None`, `task_payload` must also be specified.
    task_payload : dict, default = None
        If specified, adds the specified task to the `ControlMessage`. This parameter is only valid when `message_type`
        is set to `CONTROL_MESSAGE`. If not `None`, `task_type` must also be specified.
    """

    def __init__(self,
                 config: Config,
                 message_type: SupportedMessageTypes = SupportedMessageTypes.MESSAGE_META,
                 task_type: str = None,
                 task_payload: dict = None):
        super().__init__(config)

        self._message_type = message_type
        self._task_type = task_type
        self._task_payload = task_payload

        if (self._message_type is SupportedMessageTypes.CONTROL_MESSAGE):
            if ((self._task_type is None) != (self._task_payload is None)):
                raise ValueError("Both `task_type` and `task_payload` must be specified if either is specified.")
        elif (self._message_type is SupportedMessageTypes.MESSAGE_META):
            if (self._task_type is not None or self._task_payload is not None):
                raise ValueError("Cannot specify `task_type` or `task_payload` for non-control messages.")
        else:
            raise ValueError(f"Invalid message type: {self._message_type}")

    def compute_schema(self, schema: _pipeline.StageSchema):
        if (self._message_type is SupportedMessageTypes.CONTROL_MESSAGE):
            schema.output_schema.set_type(ControlMessage)
        else:
            schema.output_schema.set_type(MessageMeta)
