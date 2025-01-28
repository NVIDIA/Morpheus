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
from morpheus.utils.module_ids import FILTER_CM_FAILED
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)


@register_module(FILTER_CM_FAILED, MORPHEUS_MODULE_NAMESPACE)
def filter_cm_failed(builder: mrc.Builder):
    """
    This module discards control message if "cm_failed" field is set to True.


    Parameters
    ----------
    builder: mrc.Builder
        An mrc Builder object.

    Notes
    -----
        Configurable Parameters:
            - enable_data_type_filtering (bool): Enables filtering based on data type; Example: true; Default: false
            - on_cm_failuer (str): Enables customized configuration for operations on failed CM
    """

    config = builder.get_current_module_config()
    on_cm_failure = config.get("on_cm_failure", None)

    # pylint: disable=inconsistent-return-statements
    def on_data(control_message: ControlMessage):
        if control_message.has_metadata("cm_failed"):
            cm_failed = control_message.get_metadata("cm_failed")
            if cm_failed == "true":
                if control_message.has_metadata("cm_failed_reason"):
                    cm_failed_reason = control_message.get_metadata("cm_failed_reason")
                    logger.error("cm_failed: true, cm_failed_reason: %s", cm_failed_reason)
                else:
                    logger.error("cm_failed: true, cm_failed_reason: None")

                # Note: support customized operations
                if on_cm_failure:
                    logger.debug("on_cm_failure: %s", on_cm_failure)

                return

        return control_message

    node = builder.make_node(FILTER_CM_FAILED, ops.map(on_data))

    # Register input and output port for a module.
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
