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

from morpheus.config import Config
from morpheus.pipeline.boundary_stage_mixin import BoundaryStageMixin
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema

logger = logging.getLogger(__name__)


class LinearBoundaryEgressStage(BoundaryStageMixin, PassThruTypeMixin, GpuAndCpuMixin, SinglePortStage):
    """
    The LinearBoundaryEgressStage acts as an egress point from one linear segment to another. Given an existing linear
    pipeline that we want to connect to another segment, a linear boundary egress stage would be added, in conjunction
    with a matching LinearBoundaryIngressStage on the target linear segment.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    boundary_port_id : `str`
        String indicating the name of the egress port associated with the LinearBoundaryEgressStage; allowing it to be
        paired with the corresponding ingress port.
    data_type : `typing.Type`
        Data type that this Stage will accept and then output to its egress port.

    Examples
    --------
    >>> boundary_egress = LinearBoundaryEgressStage(config, "my_boundary_port", int)
    """

    def __init__(self, c: Config, boundary_port_id: str, data_type):
        super().__init__(c)

        self._port_id = boundary_port_id
        self._output_type = data_type if data_type else typing.Any

    @property
    def name(self) -> str:
        return "linear_segment_egress"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple
            Accepted input types.

        """
        return (self._output_type, )

    def supports_cpp_node(self):
        return False

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        boundary_egress = builder.get_egress(self._port_id)
        builder.make_edge(input_node, boundary_egress)

        return input_node


class LinearBoundaryIngressStage(BoundaryStageMixin, PreallocatorMixin, GpuAndCpuMixin, SingleOutputSource):
    """
    The LinearBoundaryIngressStage acts as source ingress point from a corresponding egress in another linear segment.
    Given an existing linear pipeline that we want to connect to another segment, a linear boundary egress stage would
    be added to it and a matching LinearBoundaryIngressStage would be created to receive the egress point.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    boundary_port_id : `str`
        String corresponding to the name of the LinearBoundaryEgressStage's egress port, use to identify and connect
        with the appropriate egress.
    data_type : `object`
        Data type that this Stage will accept, which will correspond to some existing egress output.

    Examples
    --------
    >>> boundary_ingress = LinearBoundaryIngressStage(config, "my_boundary_port", int)
    """

    def __init__(self, c: Config, boundary_port_id: str, data_type=None):
        super().__init__(c)

        self._port_id = boundary_port_id
        self._output_type = data_type if data_type else typing.Any

    @property
    def name(self) -> str:
        return "segment_boundary_ingress"

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(self._output_type)

    def supports_cpp_node(self):
        return False

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:
        boundary_ingress = builder.get_ingress(self._port_id)
        source = builder.make_node(self.unique_name, ops.map(lambda data: data))
        builder.make_edge(boundary_ingress, source)

        return source
