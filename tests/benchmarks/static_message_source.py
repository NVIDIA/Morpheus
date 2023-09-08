# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import mrc.core.operators as ops
import typing_utils

import cudf

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline import SingleOutputSource
from morpheus.pipeline import StreamPair

# TODO: test benchmarks, as far as I can tell the post-flatten node was not used.


class StaticMessageSource(SingleOutputSource):

    def __init__(self, c: Config, df: cudf.DataFrame):
        super().__init__(c)

        self._batch_size = c.pipeline_batch_size
        self._df = df

    @property
    def name(self) -> str:
        return "static-data"

    def supports_cpp_node(self):
        return False

    @property
    def input_count(self) -> int:
        return len(self._df)

    def output_type(self) -> type:
        return MessageMeta

    def _build_source(self, builder: mrc.Builder) -> StreamPair:
        out_stream = builder.make_source(self.unique_name, self._generate_frames())
        return out_stream, MessageMeta

    def _generate_frames(self):
        yield MessageMeta(self._df)
