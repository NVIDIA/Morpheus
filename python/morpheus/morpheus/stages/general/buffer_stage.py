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

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.logger import deprecated_stage_warning

logger = logging.getLogger(__name__)


@register_stage("buffer", command_args={"deprecated": True})
class BufferStage(PassThruTypeMixin, SinglePortStage):
    """
    Buffer results.

    The input messages are buffered by this stage class for faster access to downstream stages. Allows
    upstream stages to run faster than downstream stages.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self, c: Config, count: int = 1000):
        super().__init__(c)

        self._buffer_count = count

    @property
    def name(self) -> str:
        return "buffer"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple
            Accepted input types.

        """
        return (typing.Any, )

    def supports_cpp_node(self):
        return False

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        reason = "The stage is no longer required to manage backpressure and has been deprecated. It has no" \
                 " effect acts as a pass through stage."
        deprecated_stage_warning(logger, type(self), self.unique_name, reason=reason)

        return input_node
