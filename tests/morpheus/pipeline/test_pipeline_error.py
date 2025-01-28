#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import collections.abc
import time

import mrc
import pytest

from morpheus.config import Config
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.pipeline.stage_decorator import source
from morpheus.pipeline.stage_decorator import stage


@source
def error_source(subscription: mrc.Subscription, *, raise_error: bool = False) -> collections.abc.Iterator[int]:
    yield 1

    if raise_error:
        raise RuntimeError("Test error in source")

    while subscription.is_subscribed():
        time.sleep(0.1)


@stage
def error_stage(i: int, *, raise_error: bool = False) -> int:
    if raise_error:
        raise RuntimeError("Test error in stage")

    return i


@pytest.mark.parametrize("source_error, stage_error", [(True, False), (False, True), (True, True)])
def test_pipeline(config: Config, source_error: bool, stage_error: bool):
    """
    When source_error=False and stage_error=True this reproduces issue #1838
    """

    config = Config()
    pipe = LinearPipeline(config)
    pipe.set_source(error_source(config, raise_error=source_error))
    pipe.add_stage(error_stage(config, raise_error=stage_error))

    with pytest.raises(RuntimeError, match="^Test error in (source|stage)$"):
        pipe.run()
