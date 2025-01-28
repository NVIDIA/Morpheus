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
"""Postprocessing stage for Digital Fingerprinting pipeline."""

import logging
import time
import typing
from datetime import datetime

import mrc
from mrc.core import operators as ops

from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage

logger = logging.getLogger(f"morpheus.{__name__}")


class DFPPostprocessingStage(PassThruTypeMixin, SinglePortStage):
    """
    This stage adds a new `event_time` column to the DataFrame indicating the time which Morpheus detected the
    anomalous messages, and replaces any `NAN` values with the a string value of `'NaN'`.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    """

    def __init__(self, c: Config):
        super().__init__(c)
        self._needed_columns['event_time'] = TypeId.STRING

    @property
    def name(self) -> str:
        """Stage name."""
        return "dfp-postproc"

    def supports_cpp_node(self):
        """Whether this stage supports a C++ node."""
        return False

    def accepted_types(self) -> typing.Tuple:
        """Accepted input types."""
        return (ControlMessage, )

    def _process_events(self, message: ControlMessage):
        # Assume that a filter stage preceedes this stage
        with message.payload().mutable_dataframe() as df:
            df['event_time'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

    def on_data(self, message: ControlMessage):
        """Process a message."""
        if (not message or message.payload().count == 0):
            return None

        start_time = time.time()

        self._process_events(message)

        duration = (time.time() - start_time) * 1000.0

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Completed postprocessing for user %s in %s ms. Event count: %s. Start: %s, End: %s",
                         message.get_metadata("user_id"),
                         duration,
                         message.payload().count,
                         message.payload().get_data(self._config.ae.timestamp_column_name).min(),
                         message.payload().get_data(self._config.ae.timestamp_column_name).max())

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data), ops.filter(lambda x: x is not None))
        builder.make_edge(input_node, node)

        return node
