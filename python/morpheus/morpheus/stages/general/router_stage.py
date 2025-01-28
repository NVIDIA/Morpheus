# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
import mrc.core.segment

import morpheus.pipeline as _pipeline  # pylint: disable=cyclic-import
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin

logger = logging.getLogger(__name__)


@register_stage("router")
class RouterStage(GpuAndCpuMixin, _pipeline.Stage):
    """
    Buffer results.

    The input messages are buffered by this stage class for faster access to downstream stages. Allows upstream stages
    to run faster than downstream stages.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    keys : `list[str]`
        List of keys to route the messages.
    key_fn : `typing.Callable[[object], str]`
        Function to determine the key for the message. The function should take a message as input and return a key. The
        key should be one of the keys in the `keys` list.
    processing_engines : `int`
        Number of processing engines to use for the router. If set to 0, the router will use the thread from the
        upstream node for processing. In this situation, slow downstream nodes can block which can prevent routing to
        other, non-blocked downstream nodes. To resolve this, set the `processing_engines` parameter to a value greater
        than 0. This will create separate engines (similar to a thread) which can continue routing even if one gets
        blocked. Higher values of `processing_engines` can prevent blocking at the expense of additional threads.

    """

    def __init__(self,
                 c: Config,
                 *,
                 keys: list[str],
                 key_fn: typing.Callable[[object], str],
                 processing_engines: int = 0) -> None:
        super().__init__(c)

        self._keys = keys
        self._key_fn = key_fn
        self._processing_engines = processing_engines

        if (self._processing_engines < 0):
            raise ValueError("Invalid number of processing engines. Must be greater than or equal to 0.")

        if (len(keys) == 0):
            raise ValueError("Router stage must have at least one key.")

        self._router: mrc.core.segment.SegmentObject | None = None

        self._create_ports(1, len(keys))

    @property
    def name(self) -> str:
        return "router"

    def supports_cpp_node(self):
        return True

    def compute_schema(self, schema: _pipeline.StageSchema):

        # Get the input type
        input_type = schema.input_type

        for port_idx in range(len(self._keys)):
            schema.output_schemas[port_idx].set_type(input_type)

    def _build(self, builder: mrc.Builder, input_nodes: list[mrc.SegmentObject]) -> list[mrc.SegmentObject]:

        assert len(input_nodes) == 1, "Router stage should have exactly one input node"

        from mrc.core.node import Router
        from mrc.core.node import RouterComponent

        if (self._processing_engines > 0):
            self._router = Router(builder, self.unique_name, router_keys=self._keys, key_fn=self._key_fn)

            self._router.launch_options.engines_per_pe = self._processing_engines
        else:
            self._router = RouterComponent(builder, self.unique_name, router_keys=self._keys, key_fn=self._key_fn)

        builder.make_edge(input_nodes[0], self._router)

        return [self._router.get_child(k) for k in self._keys]
