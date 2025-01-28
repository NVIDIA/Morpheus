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

import os

from _utils import TEST_DIRS
from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.stages.input.file_source_stage import FileSourceStage


def test_execution_modes(config: Config):
    assert issubclass(FileSourceStage, GpuAndCpuMixin)
    stage = FileSourceStage(config, filename=os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv"))

    # we don't care about the order of the execution modes
    assert set(stage.supported_execution_modes()) == {ExecutionMode.GPU, ExecutionMode.CPU}
