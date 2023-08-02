#!/usr/bin/env python
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

import gc
import typing

import pytest

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils.type_aliases import DataFrameType


class SourceTestStage(InMemorySourceStage):

    def __init__(self,
                 config,
                 dataframes: typing.List[DataFrameType],
                 state_dict: dict,
                 repeat: int = 1,
                 state_key: str = "source"):
        super().__init__(config, dataframes, repeat)
        self._state_dict = state_dict
        self._state_key = state_key

    @property
    def name(self) -> str:
        return "test-source"

    def __del__(self):
        self._state_dict[self._state_key] = True
        self._state_dict = None


class SinkTestStage(InMemorySinkStage):

    def __init__(self, config, state_dict: dict, state_key: str = "sink"):
        super().__init__(config)
        self._state_dict = state_dict
        self._state_key = state_key

    @property
    def name(self) -> str:
        return "test-sink"

    def __del__(self):
        self._state_dict[self._state_key] = True
        self._state_dict = None


def _run_pipeline(config: Config, filter_probs_df: DataFrameType, state_dict: dict):
    pipe = LinearPipeline(config)
    pipe.set_source(SourceTestStage(config, [filter_probs_df], state_dict=state_dict))
    pipe.add_stage(SinkTestStage(config, state_dict=state_dict))
    pipe.run()


@pytest.mark.use_cudf
def test_destructors_called(config: Config, filter_probs_df: DataFrameType):
    """
    Test to ensure that the destructors of stages are called (issue #1114).
    """
    state_dict = {"source": False, "sink": False}
    _run_pipeline(config, filter_probs_df, state_dict)

    gc.collect()
    assert state_dict["source"]
    assert state_dict["sink"]
