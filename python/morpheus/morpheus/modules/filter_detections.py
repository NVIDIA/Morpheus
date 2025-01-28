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
import pickle

import mrc
from mrc.core import operators as ops

import morpheus._lib.stages as _stages
from morpheus.common import FilterSource
from morpheus.controllers.filter_detections_controller import FilterDetectionsController
from morpheus.utils.module_ids import FILTER_DETECTIONS
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)


@register_module(FILTER_DETECTIONS, MORPHEUS_MODULE_NAMESPACE)
def filter_detections(builder: mrc.Builder):
    """
    Filter message by a classification threshold.

    The FilterDetections is used to filter rows from a dataframe based on values in a tensor using a specified
    criteria. Rows in the `meta` dataframe are excluded if their associated value in the `probs` array is less than or
    equal to `threshold`.

    This module can operate in two different modes set by the `copy` argument.
    When the `copy` argument is `True` (default), rows that meet the filter criteria are copied into a new dataframe.
    When `False` sliced views are used instead.

    Setting `copy=True` should be used when the number of matching records is expected to be both high and in
    non-adjacent rows. In this mode, the stage will generate only one output message for each incoming message,
    regardless of the size of the input and the number of matching records. However this comes at the cost of needing to
    allocate additional memory and perform the copy.
    Note: In most other stages, messages emitted contain a reference to the original `MessageMeta` emitted into the
    pipeline by the source stage. When using copy mode this won't be the case and could cause the original `MessageMeta`
    to be deallocated after this stage.

    Setting `copy=False` should be used when either the number of matching records is expected to be very low or are
    likely to be contained in adjacent rows. In this mode, slices of contiguous blocks of rows are emitted in multiple
    output messages. Performing a slice is relatively low-cost, however for each incoming message the number of emitted
    messages could be high (in the worst case scenario as high as half the number of records in the incoming message).
    Depending on the downstream stages, this can cause performance issues, especially if those stages need to acquire
    the Python GIL.

    Parameters
    ----------
    builder : mrc.Builder
        An mrc Builder object.

    Notes
    -----
        Configurable Parameters:
            - copy (bool): Whether to copy the rows or slice them; Example: true; Default: true
            - field_name (str): Name of the field to filter on; Example: `probs`; Default: probs
            - filter_source (str): Source of the filter field; Example: `AUTO`; Default: AUTO
            - schema (dict): Schema configuration; See Below; Default: -
            - threshold (float): Threshold value to filter on; Example: 0.5; Default: 0.5

        schema:
            - encoding (str): Encoding; Example: "latin1"; Default: "latin1"
            - input_message_type (str): Pickled message type; Example: `pickle_message_type`; Default: `[Required]`
            - schema_str (str): Schema string; Example: "string"; Default: `[Required]`
    """

    config = builder.get_current_module_config()

    field_name = config.get("field_name", "probs")
    threshold = config.get("threshold", 0.5)
    filter_source = config.get("filter_source", "AUTO")
    use_cpp = config.get("use_cpp", True)

    filter_source_dict = {"AUTO": FilterSource.Auto, "DATAFRAME": FilterSource.DATAFRAME, "TENSOR": FilterSource.TENSOR}

    copy = config.get("copy", True)

    if ("schema" not in config):
        raise ValueError("Schema configuration not found.")

    schema_config = config["schema"]
    input_message_type = schema_config["input_message_type"]
    encoding = schema_config["encoding"]

    message_type = pickle.loads(bytes(input_message_type, encoding))

    controller = FilterDetectionsController(threshold=threshold,
                                            filter_source=filter_source_dict[filter_source],
                                            field_name=field_name)

    controller.update_filter_source(message_type=message_type)

    if use_cpp:
        node = _stages.FilterDetectionsStage(builder,
                                             FILTER_DETECTIONS,
                                             controller.threshold,
                                             copy,
                                             controller.filter_source,
                                             controller.field_name)
    else:
        if copy:
            node = builder.make_node(FILTER_DETECTIONS,
                                     ops.map(controller.filter_copy),
                                     ops.filter(lambda x: x is not None))
        else:
            # Convert list returned by `filter_slice` back to individual messages
            node = builder.make_node(FILTER_DETECTIONS, ops.map(controller.filter_slice), ops.flatten())

    # Register input and output port for a module.
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
