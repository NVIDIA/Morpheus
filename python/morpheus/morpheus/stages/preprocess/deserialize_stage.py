# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.modules.preprocess.deserialize import DeserializeLoaderFactory
from morpheus.pipeline.control_message_stage import ControlMessageStage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.stage_schema import StageSchema

logger = logging.getLogger(__name__)


@register_stage("deserialize",
                modes=[PipelineModes.FIL, PipelineModes.NLP, PipelineModes.OTHER],
                ignore_args=["task_type", "task_payload"])
class DeserializeStage(GpuAndCpuMixin, ControlMessageStage):
    """
    Messages are logically partitioned based on the pipeline config's `pipeline_batch_size` parameter.

    This stage deserialize the output of `FileSourceStage`/`KafkaSourceStage` into a `ControlMessage`. This
    should be one of the first stages after the `Source` object.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    ensure_sliceable_index : bool, default = True
        Whether or not to call `ensure_sliceable_index()` on all incoming `MessageMeta`, which will replace the index
        of the underlying dataframe if the existing one is not unique and monotonic.
    task_type : str, default = None
        If specified, adds the specified task to the `ControlMessage`. If not `None`, `task_payload` must also be
        specified.
    task_payload : dict, default = None
        If specified, adds the specified task to the `ControlMessage`. If not `None`, `task_type` must also be
        specified.
    """

    def __init__(self,
                 c: Config,
                 *,
                 ensure_sliceable_index: bool = True,
                 task_type: str = None,
                 task_payload: dict = None):
        super().__init__(c)

        self._batch_size = c.pipeline_batch_size
        self._ensure_sliceable_index = ensure_sliceable_index

        self._max_concurrent = c.num_threads

        # Mark these stages to log timestamps if requested
        self._should_log_timestamps = True

        self._task_type = task_type
        self._task_payload = task_payload

        if ((self._task_type is None) != (self._task_payload is None)):
            raise ValueError("Both `task_type` and `task_payload` must be specified if either is specified.")

    @property
    def name(self) -> str:
        return "deserialize"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        """
        return (MessageMeta, )

    def supports_cpp_node(self):
        # Enable support by default
        return True

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(ControlMessage)

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        if (self._build_cpp_node()):
            import morpheus._lib.stages as _stages
            out_node = _stages.DeserializeStage(builder,
                                                self.unique_name,
                                                batch_size=self._batch_size,
                                                ensure_sliceable_index=self._ensure_sliceable_index,
                                                task_type=self._task_type,
                                                task_payload=self._task_payload)

            builder.make_edge(input_node, out_node)
        else:
            module_config = {
                "ensure_sliceable_index": self._ensure_sliceable_index,
                "task_type": self._task_type,
                "task_payload": self._task_payload,
                "batch_size": self._batch_size,
                "max_concurrency": self._max_concurrent,
                "should_log_timestamp": self._should_log_timestamps
            }

            module_loader = DeserializeLoaderFactory.get_instance(module_name=f"deserialize_{self.unique_name}",
                                                                  module_config=module_config)

            module = module_loader.load(builder=builder)
            mod_in_node = module.input_port("input")
            out_node = module.output_port("output")

            builder.make_edge(input_node, mod_in_node)

        return out_node
