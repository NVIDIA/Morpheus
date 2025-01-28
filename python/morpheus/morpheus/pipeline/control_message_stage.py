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
import time

import mrc

import morpheus.pipeline as _pipeline  # pylint: disable=cyclic-import
from morpheus.config import Config
from morpheus.messages import ControlMessage

logger = logging.getLogger(__name__)


def _get_time_ms():
    return round(time.time() * 1000)


class ControlMessageStage(_pipeline.SinglePortStage):
    """
    Subclass of SinglePortStage with option to log timestamps in MessageMeta dataframe.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    """

    def __init__(self, c: Config):

        # Derived classes should set this to true to log timestamps in debug builds
        self._should_log_timestamps = False

        super().__init__(c)

    def compute_schema(self, schema: _pipeline.StageSchema):
        schema.output_schema.set_type(ControlMessage)

    def _post_build_single(self, builder: mrc.Builder, out_node: mrc.SegmentObject) -> mrc.SegmentObject:

        # Check if we are debug and should log timestamps.
        if (self._config.debug and self._should_log_timestamps):
            # Cache the name property. Removes dependency on self in callback
            cached_name = self.name

            logger.info("Adding timestamp info for stage: '%s'", cached_name)

            def post_timestamps(message: ControlMessage):

                curr_time = _get_time_ms()

                message.set_timestamp("_ts_" + cached_name, str(curr_time))

                # Must return the original object
                return message

            # Only have one port
            post_ts = builder.make_node(self.unique_name + "-ts", post_timestamps)
            builder.make_edge(out_node, post_ts)
            out_node = post_ts

        return super()._post_build_single(builder, out_node)
