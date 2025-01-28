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

import typing

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.atomic_integer import AtomicInteger


@register_stage("unittest-dfp-length-check")
class DFPLengthChecker(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Verifies that the incoming MessageMeta classes are of a specific length

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    expected_length : int
        Expected length of incoming messages

    num_exact: int
        Number of messages to check. For a datasource with 2000 records if we expect the first message to be of lenth
        1024, the next message will contain only 976. The first `num_exact` messages will be checked with `==` and `<=`
        after that.
    """

    def __init__(self, c: Config, expected_length: int, num_exact: int = 1):
        super().__init__(c)

        self._expected_length = expected_length
        self._num_exact = num_exact
        self._num_received = AtomicInteger(0)

    @property
    def name(self) -> str:
        return "dfp-length-check"

    def accepted_types(self) -> typing.Tuple:
        return (MessageMeta)

    def supports_cpp_node(self) -> bool:
        return False

    def _length_checker(self, x: MessageMeta) -> MessageMeta:
        msg_num = self._num_received.get_and_inc()
        if msg_num < self._num_exact:
            assert x.count == self._expected_length, \
                f"Unexpected number of rows in message number {msg_num}: {x.count} != {self._num_exact}"
        else:
            assert x.count <= self._expected_length, \
                f"Unexpected number of rows in message number {msg_num}: {x.count} > {self._num_exact}"

        return x

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self._length_checker))
        builder.make_edge(input_node, node)

        return node
