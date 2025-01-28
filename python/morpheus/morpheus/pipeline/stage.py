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

import mrc

import morpheus.pipeline as _pipeline  # pylint: disable=cyclic-import

logger = logging.getLogger(__name__)


class Stage(_pipeline.StageBase):
    """
    This class serves as the base for all pipeline stage implementations that are not source objects.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def _post_build(self, builder: mrc.Builder, out_ports_nodes: list[mrc.SegmentObject]) -> list[mrc.SegmentObject]:

        return out_ports_nodes

    def _start(self):
        pass

    def _on_complete(self, node):  # pylint: disable=unused-argument
        logger.info("Stage Complete: %s", self.name)
