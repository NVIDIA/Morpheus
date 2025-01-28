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

logger = logging.getLogger(__name__)


class SourceStage(_pipeline.StageBase):
    """
    The SourceStage is mandatory for the Morpheus pipeline to run. This stage represents the start of the pipeline. All
    `SourceStage` object take no input but generate output.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._start_callbacks: typing.List[typing.Callable] = []
        self._stop_callbacks: typing.List[typing.Callable] = []

        self._sources: list[mrc.SegmentObject] = []

    @property
    def input_count(self) -> int:
        """
        Return None for no max intput count.
        """
        return None

    @abstractmethod
    def _build_sources(self, builder: mrc.Builder) -> list[mrc.SegmentObject]:
        """
        Abstract method all derived Source classes should implement. Returns the same value as `build`.

        :meta public:

        Returns
        -------

        `mrc.SegmentObject`:
            The MRC nodes for this stage.
        """

        pass

    @typing.final
    def _pre_build(self, do_propagate: bool = True):
        assert len(self.input_ports) == 0, "Sources shouldnt have input ports"
        return super()._pre_build(do_propagate=do_propagate)

    @typing.final
    def _build(self, builder: mrc.Builder, input_nodes: list[mrc.SegmentObject]) -> list[mrc.SegmentObject]:
        # Derived source stages should override `_build_source` instead of this method. This allows for tracking the
        # True source object separate from the output node. If any other operators need to be added after the source,
        # use `_post_build`
        assert len(self.input_ports) == 0, "Sources shouldnt have input ports"
        assert len(input_nodes) == 0, "Sources shouldnt have input nodes"

        sources = self._build_sources(builder)

        assert len(sources) == len(self.output_ports), "Number of sources should match number of output ports"

        for (i, source) in enumerate(sources):
            self._output_ports[i]._output_node = source
            self._sources.append(source)

        return sources

    def _post_build(self, builder: mrc.Builder, out_ports_nodes: list[mrc.SegmentObject]) -> list[mrc.SegmentObject]:

        return out_ports_nodes

    def _start(self):
        for source in self._sources:
            source.start()

    async def join(self):
        """
        Awaitable method that stages can implement this to perform cleanup steps when pipeline is stopped.
        Typically this is called after `stop` during a graceful shutdown, but may not be called if the pipeline is
        terminated on its own.
        """
        pass
