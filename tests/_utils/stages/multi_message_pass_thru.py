# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Simple pass-thru stages for the purposes of testing PassThruTypeMixin and InferredPassThruTypeMixin"""

import mrc
from mrc.core import operators as ops

from morpheus.messages import MultiMessage
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair


class BasePassThruStage(SinglePortStage):

    def accepted_types(self) -> (MultiMessage, ):
        return (MultiMessage, )

    def supports_cpp_node(self) -> bool:
        return False

    def on_data(self, message: MultiMessage):
        return message

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_stream[0], node)

        return node, input_stream[1]


class MultiMessagePassThruStage(PassThruTypeMixin, BasePassThruStage):

    @property
    def name(self) -> str:
        return "mm-pass-thru"


class InferredMultiMessagePassThruStage(PassThruTypeMixin, BasePassThruStage):

    @property
    def name(self) -> str:
        return "inf-mm-pass-thru"
