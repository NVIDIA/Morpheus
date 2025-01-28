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
from abc import abstractmethod

import mrc

import morpheus.pipeline as _pipeline  # pylint: disable=cyclic-import
from morpheus.config import Config
from morpheus.utils.type_utils import pretty_print_type_name

logger = logging.getLogger(__name__)


class SingleOutputSource(_pipeline.SourceStage):
    """
    Subclass of SourceStage for building source stages that generate output for single port.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._create_ports(0, 1)

        # Flag to indicate if we need to stop, subclasses should check this value periodically, typically at the start
        # of a polling loop
        self._stop_requested = False

    # pylint: disable=unused-argument
    def _post_build_single(self, builder: mrc.Builder, out_node: mrc.SegmentObject) -> mrc.SegmentObject:
        return out_node

    # pylint: enable=unused-argument

    @abstractmethod
    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:
        """
        Abstract method all derived Source classes should implement. Returns the same value as `build`.

        :meta public:

        Returns
        -------

        `mrc.SegmentObject`:
            The MRC node for this stage.
        """
        pass

    def _build_sources(self, builder: mrc.Builder) -> list[mrc.SegmentObject]:
        assert len(self.output_ports) == 1, \
            f"SingleOutputSource should have only one output port, {self} has {len(self.output_ports)}"

        return [self._build_source(builder)]

    @typing.final
    def _post_build(self, builder: mrc.Builder, out_ports_nodes: list[mrc.SegmentObject]) -> list[mrc.SegmentObject]:
        ret_val = self._post_build_single(builder, out_ports_nodes[0])

        logger.info("Added source: %s\n  └─> %s", self, pretty_print_type_name(self.output_ports[0].output_type))

        return [ret_val]

    def stop(self):
        """
        This method is invoked by the pipeline whenever there is an unexpected shutdown.
        Subclasses should override this method to perform any necessary cleanup operations.
        """

        # Indicate we need to stop
        self.request_stop()

        return super().stop()

    def request_stop(self):
        """
        Request the source to stop processing data.
        """
        self._stop_requested = True

    def is_stop_requested(self) -> bool:
        """
        Returns `True` if a stop has been requested.

        Returns
        -------
        bool:
            True if a stop has been requested, False otherwise.
        """
        return self._stop_requested
