#!/usr/bin/env python
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

import mrc
from mrc.core import operators as ops

import cudf

from morpheus.common import typeid_to_numpy_str
from morpheus.messages import ControlMessage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage


class CheckPreAlloc(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Acts like add-class/add-scores in that it requests a preallocation, the node will assert that the preallocation
    occurred with the correct type.
    """

    def __init__(self, c, probs_type):
        super().__init__(c)
        self._expected_type = cudf.dtype(typeid_to_numpy_str(probs_type))
        self._class_labels = c.class_labels
        self._needed_columns.update({label: probs_type for label in c.class_labels})

    @property
    def name(self) -> str:
        return "check-prealloc"

    def accepted_types(self) -> tuple:
        return (ControlMessage, )

    def supports_cpp_node(self) -> bool:
        return False

    def _check_prealloc(self, msg: ControlMessage) -> ControlMessage:
        df = msg.payload().df
        for label in self._class_labels:
            assert label in df.columns
            assert df[label].dtype == self._expected_type

        return msg

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self._check_prealloc))
        builder.make_edge(input_node, node)

        return node
