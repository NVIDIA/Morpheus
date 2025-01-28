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


@register_stage("delay", command_args={"deprecated": True})
class DelayStage(PassThruTypeMixin, SinglePortStage):
    """
    Delay results for a certain duration.

    Delay stage class. Used to buffer all inputs until the timeout duration is hit. At that point all messages
    will be dumped into downstream stages. Useful for testing performance of one stage at a time.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self, c: Config, duration: str):
        super().__init__(c)

        self._duration = duration

    @property
    def name(self) -> str:
        return "delay"

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
