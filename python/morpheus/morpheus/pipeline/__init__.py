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
"""
All objects related to building and running a pipeline.
"""

# Note: The pipeline module is unique in that we re-export all of the classes and functions from the submodules. To
# avoid circular imports, we must import the classes in a specific order. And in each submodule, we should never import
# the from pipeline submodules. Instead, we should import from the parent module as a namespace package and then use the
# fully qualified name to access the classes. For example, in morpheus/pipeline/stage.py:
# Do not do this:
# ```
# from morpheus.pipeline.stage_base import StageBase
# ```
# Instead, do this:
# ```
# import morpheus.pipeline as _pipeline  # pylint: disable=cyclic-import
# class Stage(_pipeline.StageBase):
# ```

# These must be imported in a specific order
# isort: off

from morpheus.pipeline.boundary_stage_mixin import BoundaryStageMixin
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.stage_schema import PortSchema
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.pipeline.sender import Sender
from morpheus.pipeline.receiver import Receiver
from morpheus.pipeline.stage_base import StageBase
from morpheus.pipeline.stage import Stage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.control_message_stage import ControlMessageStage
from morpheus.pipeline.source_stage import SourceStage
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.configurable_output_source import ConfigurableOutputSource
from morpheus.pipeline.pipeline import Pipeline
from morpheus.pipeline.linear_pipeline import LinearPipeline
