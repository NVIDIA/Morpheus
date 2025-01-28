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
from datetime import timedelta

import mrc

from morpheus.cli import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import MessageMeta
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema

logger = logging.getLogger(__name__)

MAX_PKT_RECEIVE = 512 * 16


@register_stage("from-doca-convert", modes=[PipelineModes.NLP])
class DocaConvertStage(PreallocatorMixin, SinglePortStage):
    """
    A source stage used to receive raw packet data from a ConnectX-6 Dx NIC.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    max_batch_delay_sec : `float`
        Maximum amount of time to wait, in seconds, for additional incoming packets prior to constructing a cuDF
        DataFrame.
    max_batch_size : `int`
        Maximum number of packets to attempt to combine into a single cuDF DataFrame. Must be greater than or equal to
        `MAX_PKT_RECEIVE`.
    buffer_channel_size : `int`, optional
        The size of the internal buffer to store incoming packet data. If `None`, the config's `edge_buffer_size` will
        be used.
    """

    def __init__(self,
                 c: Config,
                 max_batch_delay_sec: float = 0.5,
                 max_batch_size: int = MAX_PKT_RECEIVE * 5,
                 buffer_channel_size: int = None):

        super().__init__(c)

        self._max_batch_delay = timedelta(seconds=max_batch_delay_sec)
        self._buffer_channel_size = buffer_channel_size or c.edge_buffer_size

        if max_batch_size < MAX_PKT_RECEIVE:
            raise RuntimeError(f"max_batch_size ({max_batch_size}) must be greater than or equal to {MAX_PKT_RECEIVE}")

        self._max_batch_size = max_batch_size

        # Attempt to import the C++ stage on creation
        try:
            # pylint: disable=c-extension-no-member
            import morpheus._lib.doca as _doca

            self.doca_convert_class = _doca.DocaConvertStage
        except ImportError as ex:
            raise NotImplementedError(("The Morpheus DOCA components could not be imported. "
                                       "Ensure the DOCA components have been built and installed. Error message: ") +
                                      ex.msg) from ex

    @property
    def name(self) -> str:
        return "from-doca-convert"

    @property
    def input_count(self) -> int:
        """Return None for no max input count"""
        return None

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(MessageMeta)

    def supports_cpp_node(self):
        return True

    def accepted_types(self) -> tuple:
        return (typing.Any, )

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:

        if self._build_cpp_node():
            node = self.doca_convert_class(builder,
                                           self.unique_name,
                                           max_batch_delay=self._max_batch_delay,
                                           max_batch_size=self._max_batch_size,
                                           buffer_channel_size=self._buffer_channel_size)

            builder.make_edge(input_node, node)
            return node

        raise NotImplementedError("Does not support Python nodes")
