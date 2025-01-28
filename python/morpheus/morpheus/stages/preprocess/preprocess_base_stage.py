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

import typing

import mrc
from mrc.core import operators as ops

from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.pipeline.control_message_stage import ControlMessageStage
from morpheus.pipeline.stage_schema import StageSchema


class PreprocessBaseStage(ControlMessageStage):
    """
    This is a base pre-processing class holding general functionality for all preprocessing stages.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._should_log_timestamps = True

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        """
        return (ControlMessage, )

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(ControlMessage)

    def _get_preprocess_fn(self) -> typing.Callable[[ControlMessage], ControlMessage]:
        """
        This method should be implemented by any subclasses with a Python implementation.
        """
        raise NotImplementedError("No Python implementation provided by this stage")

    def _get_preprocess_node(self, builder: mrc.Builder) -> mrc.SegmentObject:
        """
        This method should be implemented by any subclasses with a C++ implementation.
        """
        raise NotImplementedError("No C++ implementation provided by this stage")

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        if self._build_cpp_node():
            node = self._get_preprocess_node(builder)
            node.launch_options.pe_count = self._config.num_threads
        else:
            preprocess_fn = self._get_preprocess_fn()
            node = builder.make_node(self.unique_name, ops.map(preprocess_fn))

        builder.make_edge(input_node, node)

        return node
