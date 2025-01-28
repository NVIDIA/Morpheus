# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.common import FilterSource
from morpheus.config import Config
from morpheus.controllers.filter_detections_controller import FilterDetectionsController
from morpheus.messages import ControlMessage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema

logger = logging.getLogger(__name__)


@register_stage("filter")
class FilterDetectionsStage(GpuAndCpuMixin, SinglePortStage):
    """
    Filter message by a classification threshold.

    The FilterDetectionsStage is used to filter rows from a dataframe based on values in a tensor using a specified
    criteria. Rows in the `meta` dataframe are excluded if their associated value in the `probs` array is less than or
    equal to `threshold`.

    This stage can operate in two different modes set by the `copy` argument.
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
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    threshold : float
        Threshold to classify, default is 0.5.
    copy : bool
        Whether or not to perform a copy.
    filter_source : `morpheus.common.FilterSource`, case_sensitive = False
        Indicate if we are operating on is an output tensor or a field in the DataFrame.
        Choosing `Auto` will default to `TENSOR` when the incoming message contains output tensorts and `DATAFRAME`
        otherwise.
    field_name : str
        Name of the tensor or DataFrame column to use as the filter criteria
    """

    def __init__(self,
                 c: Config,
                 threshold: float = 0.5,
                 copy: bool = True,
                 filter_source: FilterSource = FilterSource.Auto,
                 field_name: str = "probs"):
        super().__init__(c)

        self._copy = copy
        self._controller = FilterDetectionsController(threshold=threshold,
                                                      filter_source=filter_source,
                                                      field_name=field_name)

    @property
    def name(self) -> str:
        return "filter"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple[`morpheus.messages.ControlMessage`, ]
            Accepted input types.

        """
        return (ControlMessage, )

    def compute_schema(self, schema: StageSchema):
        self._controller.update_filter_source(message_type=schema.input_type)
        schema.output_schema.set_type(schema.input_type)

    def supports_cpp_node(self):
        # Enable support by default
        return True

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        if self._build_cpp_node():
            import morpheus._lib.stages as _stages
            node = _stages.FilterDetectionsStage(builder,
                                                 self.unique_name,
                                                 self._controller.threshold,
                                                 self._copy,
                                                 self._controller.filter_source,
                                                 self._controller.field_name)
        else:

            if self._copy:
                node = builder.make_node(self.unique_name,
                                         ops.map(self._controller.filter_copy),
                                         ops.filter(lambda x: x is not None))
            else:
                # Use `ops.flatten` to convert the list returned by `filter_slice` back to individual messages
                node = builder.make_node(self.unique_name, ops.map(self._controller.filter_slice), ops.flatten())

        builder.make_edge(input_node, node)

        return node
