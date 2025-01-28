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

import logging

import mrc
from mrc.core import operators as ops

from morpheus.messages import ControlMessage
from morpheus.utils.module_ids import FILTER_CONTROL_MESSAGE
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)


@register_module(FILTER_CONTROL_MESSAGE, MORPHEUS_MODULE_NAMESPACE)
def filter_control_message(builder: mrc.Builder):
    """
    This module discards control messages based on specified filtering criteria.

    Parameters
    ----------
    builder : mrc.Builder
        An mrc Builder object.

    Notes
    -----
        Configurable Parameters:
            - enable_data_type_filtering (bool): Enables filtering based on data type; Example: true; Default: false
            - enable_task_filtering (bool): Enables filtering based on task type; Example: true; Default: false
            - filter_data_type (str): The data type to be used as a filter; Example: `desired_data_type`; Default: None
            - filter_task_type (str): The task type to be used as a filter; Example: `specific_task`; Default: None
    """

    config = builder.get_current_module_config()

    enable_task_filtering = config.get("enable_task_filtering", False)
    enable_data_type_filtering = config.get("enable_data_type_filtering", False)

    filter_task_type = None
    if (enable_task_filtering):
        if ("filter_task_type" not in config):
            raise ValueError("Task filtering is enabled but no task type is specified")
        filter_task_type = config["filter_task_type"]

    filter_data_type = None
    if (enable_data_type_filtering):
        if ("filter_data_type" not in config):
            raise ValueError("Data type filtering is enabled but no data type is specified")
        filter_data_type = config["filter_data_type"]

    def on_data(control_message: ControlMessage):
        cm_data_type = control_message.get_metadata("data_type")

        if enable_task_filtering:
            # Verify if control message has expected task_type.
            # TODO(Devin): Convert this to use enum values
            task_exists = control_message.has_task(filter_task_type)

            # Dispose messages if it has no expected task and it's data_type does not matches with filter_task_type.
            if (not task_exists and filter and cm_data_type != filter):
                return None
        elif (enable_data_type_filtering):
            # Regardless of whether tasks are present, discard messages
            # if the data_type don't match the filter_task_type.
            if (filter_data_type and filter_data_type != filter):
                return None

        return control_message

    node = builder.make_node(FILTER_CONTROL_MESSAGE, ops.map(on_data), ops.filter(lambda x: x is not None))

    # Register input and output port for a module.
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
