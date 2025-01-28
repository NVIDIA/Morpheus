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

import pytest

from _utils.stages.error_raiser import ErrorRaiserStage
from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils.type_aliases import DataFrameType


@pytest.mark.gpu_and_cpu_mode
@pytest.mark.parametrize("exception_cls", [RuntimeError, ValueError, NotImplementedError])
def test_stage_raises_exception(config: Config, filter_probs_df: DataFrameType, exception_cls: type[Exception]):
    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [filter_probs_df]))
    error_raiser_stage = pipe.add_stage(ErrorRaiserStage(config, exception_cls=exception_cls))
    sink_stage = pipe.add_stage(InMemorySinkStage(config))

    with pytest.raises(exception_cls):
        pipe.run()

    # Ensure that the raised exception was from our stage and not from something else
    assert error_raiser_stage.error_raised
    assert len(sink_stage.get_messages()) == 0
