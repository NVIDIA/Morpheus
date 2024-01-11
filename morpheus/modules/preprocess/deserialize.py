# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
import warnings
from functools import partial

import logging
from typing import Any
from typing import Dict
from typing import Optional

import mrc
import mrc.core.operators as ops
from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError

from morpheus.messages import MessageMeta
from morpheus.utils.column_info import ColumnInfo
from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.column_info import RenameColumn
from morpheus.utils.module_utils import register_module, ModuleInterface
from morpheus.utils.nvt.schema_converters import create_and_attach_nvt_workflow
from morpheus.utils.schema_transforms import process_dataframe
import mrc
from mrc.core import operators as ops

from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.utils.module_utils import ModuleInterface
from morpheus.utils.module_utils import register_module


logger = logging.getLogger(__name__)

def _check_slicable_index(message: MessageMeta, ensure_sliceable_index: bool = True):
    """
    Checks and ensures that the message index is sliceable.

    Parameters
    ----------
    message : MessageMeta
        The message to check for a sliceable index.
    ensure_sliceable_index : bool, optional
        Whether to ensure the message has a sliceable index.

    Returns
    -------
    MessageMeta
        The original or modified message with a sliceable index.
    """

    if (not message.has_sliceable_index()):
        if (ensure_sliceable_index):
            old_index_name = message.ensure_sliceable_index()

            if (old_index_name):
                logger.warning(("Incoming MessageMeta does not have a unique and monotonic index. "
                                "Updating index to be unique. "
                                "Existing index will be retained in column '%s'"),
                               old_index_name)

        else:
            warnings.warn(
                "Detected a non-sliceable index on an incoming MessageMeta. "
                "Performance when taking slices of messages may be degraded. "
                "Consider setting `ensure_sliceable_index==True`",
                RuntimeWarning)

    return message


def _process_dataframe_to_multi_message(message: MessageMeta, batch_size: int,
                                        ensure_sliceable_index: bool) -> typing.List[MultiMessage]:
    """
    Processes a DataFrame into a list of MultiMessage objects.

    Parameters
    ----------
    message : MessageMeta
        The message containing the DataFrame to process.
    batch_size : int
        The size of each batch.
    ensure_sliceable_index : bool
        Whether to ensure the message has a sliceable index.

    Returns
    -------
    list of MultiMessage
        A list of MultiMessage objects.
    """

    message = _check_slicable_index(message, ensure_sliceable_index)

    full_message = MultiMessage(meta=message)

    # Now break it up by batches
    output = []

    for i in range(0, full_message.mess_count, batch_size):
        output.append(full_message.get_slice(i, min(i + batch_size, full_message.mess_count)))

    return output


def _process_dataframe_to_control_message(message: MessageMeta, batch_size: int,
                                          ensure_sliceable_index: bool,
                                          task_tuple: tuple[str, dict] | None) -> typing.List[ControlMessage]:
    """
    Processes a DataFrame into a list of ControlMessage objects.

    Parameters
    ----------
    message : MessageMeta
        The message containing the DataFrame to process.
    batch_size : int
        The size of each batch.
    ensure_sliceable_index : bool
        Whether to ensure the message has a sliceable index.
    task_tuple : tuple[str, dict] | None
        Optional task to add to the ControlMessage.

    Returns
    -------
    list of ControlMessage
        A list of ControlMessage objects.
    """

    # Because ControlMessages only have a C++ implementation, we need to import the C++ MessageMeta and use that
    # 100% of the time
    # pylint: disable=morpheus-incorrect-lib-from-import
    from morpheus._lib.messages import MessageMeta as MessageMetaCpp

    message = _check_slicable_index(message, ensure_sliceable_index)

    # Now break it up by batches
    output = []

    if (message.count > batch_size):
        df = message.df

        # Break the message meta into smaller chunks
        for i in range(0, message.count, batch_size):

            message = ControlMessage()

            message.payload(MessageMetaCpp(df=df.iloc[i:i + batch_size]))

            if (task_tuple is not None):
                message.add_task(task_type=task_tuple[0], task=task_tuple[1])

            output.append(message)
    else:
        message = ControlMessage()

        message.payload(MessageMetaCpp(message.df))

        if (task_tuple is not None):
            message.add_task(task_type=task_tuple[0], task=task_tuple[1])

        output.append(message)

    return output



class DeserializeParamContract(BaseModel):
    ensure_sliceable_index: bool = True
    message_type: str = "MultiMessage"
    task_type: Optional[str] = None
    task_payload: Optional[Dict[Any, Any]] = None
    batch_size: int = 1024
    max_concurrency: int = 1
    should_log_timestamp: bool = True

@register_module("deserialize", "morpheus")
def _deserialize(builder: mrc.Builder):
    """
    Deserializes incoming messages into either MultiMessage or ControlMessage format.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus builder instance to attach this module to.

    Notes
    -----
    The `module_config` should contain:
    - 'ensure_sliceable_index': bool, whether to ensure messages have a sliceable index.
    - 'message_type': type, the type of message to output (MultiMessage or ControlMessage).
    - 'task_type': str, optional, the type of task for ControlMessages.
    - 'task_payload': dict, optional, the payload for the task in ControlMessages.
    - 'batch_size': int, the size of batches for message processing.
    - 'max_concurrency': int, optional, the maximum concurrency for processing.
    - 'should_log_timestamp': bool, optional, whether to log timestamps.
    """

    module_config = builder.get_current_module_config()

    # Validate the module configuration using the contract
    try:
        deserializer_config = DeserializeParamContract(**module_config)
    except ValidationError as e:
        error_messages = '; '.join([f"{error['loc'][0]}: {error['msg']}" for error in e.errors()])
        log_error_message = f"Invalid deserialize configuration: {error_messages}"
        logger.error(log_error_message)
        raise ValueError(log_error_message)

    ensure_sliceable_index = deserializer_config.ensure_sliceable_index
    message_type = deserializer_config.message_type
    task_type = deserializer_config.task_type
    task_payload = deserializer_config.task_payload
    batch_size = deserializer_config.batch_size
    max_concurrency = deserializer_config.max_concurrency
    should_log_timestamp = deserializer_config.should_log_timestamp

    if (task_type is not None) != (task_payload is not None):
        raise ValueError("task_type and task_payload must be both specified or both None")

    if (task_type is not None or task_payload is not None) and message_type != ControlMessage:
        raise ValueError("task_type and task_payload can only be specified for ControlMessage")

    if (message_type == "MultiMessage"):
        map_func = partial(_process_dataframe_to_multi_message,
                           batch_size=batch_size,
                           ensure_sliceable_index=ensure_sliceable_index)
    else:
        if (task_type is not None and task_payload is not None):
            task_tuple = (task_type, task_payload)
        else:
            task_tuple = None

        map_func = partial(_process_dataframe_to_control_message,
                           batch_size=batch_size,
                           ensure_sliceable_index=ensure_sliceable_index,
                           task_tuple=task_tuple)

    node = builder.make_node("deserialize", ops.map(map_func), ops.flatten(),
                             ops.filter(lambda message: message is not None))

    builder.register_module_input("input", node)
    builder.register_module_output("output", node)


DeserializeInterface = ModuleInterface("deserialize", "morpheus")
