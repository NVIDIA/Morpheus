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

import logging
import typing
import warnings
from functools import partial

import mrc
from mrc.core import operators as ops
from pydantic import ValidationError

from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.modules.schemas.deserialize_schema import DeserializeSchema
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)

DeserializeLoaderFactory = ModuleLoaderFactory("deserialize", "morpheus")


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
    if (not message):
        return None

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


def _process_dataframe_to_control_message(message: MessageMeta,
                                          batch_size: int,
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
    message = _check_slicable_index(message, ensure_sliceable_index)

    # Now break it up by batches
    output = []

    if (message.count > batch_size):
        df = message.copy_dataframe()

        # Break the message meta into smaller chunks
        for i in range(0, message.count, batch_size):

            ctrl_msg = ControlMessage()

            ctrl_msg.payload(MessageMeta(df=df.iloc[i:i + batch_size]))

            if (task_tuple is not None):
                ctrl_msg.add_task(task_type=task_tuple[0], task=task_tuple[1])

            output.append(ctrl_msg)
    else:
        ctrl_msg = ControlMessage()

        ctrl_msg.payload(MessageMeta(message.df))

        if (task_tuple is not None):
            ctrl_msg.add_task(task_type=task_tuple[0], task=task_tuple[1])

        output.append(ctrl_msg)

    return output


@register_module("deserialize", "morpheus")
def _deserialize(builder: mrc.Builder):
    """
    Deserializes incoming messages into ControlMessage format.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus builder instance to attach this module to.

    Notes
    -----
    The `module_config` should contain:
    - 'ensure_sliceable_index': bool, whether to ensure messages have a sliceable index.
    - 'task_type': str, optional, the type of task for ControlMessages.
    - 'task_payload': dict, optional, the payload for the task in ControlMessages.
    - 'batch_size': int, the size of batches for message processing.
    - 'max_concurrency': int, optional, the maximum concurrency for processing.
    - 'should_log_timestamp': bool, optional, whether to log timestamps.
    """

    module_config = builder.get_current_module_config()

    # Validate the module configuration using the contract
    try:
        deserializer_config = DeserializeSchema(**module_config)
    except ValidationError as e:
        error_messages = '; '.join([f"{error['loc'][0]}: {error['msg']}" for error in e.errors()])
        log_error_message = f"Invalid deserialize configuration: {error_messages}"
        logger.error(log_error_message)

        raise

    ensure_sliceable_index = deserializer_config.ensure_sliceable_index
    task_type = deserializer_config.task_type
    task_payload = deserializer_config.task_payload
    batch_size = deserializer_config.batch_size
    # max_concurrency = deserializer_config.max_concurrency
    # should_log_timestamp = deserializer_config.should_log_timestamp

    if (task_type is not None) != (task_payload is not None):
        raise ValueError("task_type and task_payload must be both specified or both None")

    if (task_type is not None and task_payload is not None):
        task_tuple = (task_type, task_payload)
    else:
        task_tuple = None

    map_func = partial(_process_dataframe_to_control_message,
                       batch_size=batch_size,
                       ensure_sliceable_index=ensure_sliceable_index,
                       task_tuple=task_tuple)

    node = builder.make_node("deserialize",
                             ops.map(map_func),
                             ops.flatten(),
                             ops.filter(lambda message: message is not None))

    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
