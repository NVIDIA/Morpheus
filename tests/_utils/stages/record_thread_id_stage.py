# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import threading
import typing

import mrc

from morpheus.config import Config
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage


class RecordThreadIdStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Forwarding stage that records the thread id of the progress engine
    """

    def __init__(self, config: Config):
        super().__init__(config)

        self.thread_id = None

    @property
    def name(self):
        return "record-thread"

    def accepted_types(self):
        return (typing.Any, )

    def supports_cpp_node(self):
        return False

    def _save_thread(self, x):
        self.thread_id = threading.current_thread().ident
        return x

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, mrc.core.operators.map(self._save_thread))
        builder.make_edge(input_node, node)

        return node
