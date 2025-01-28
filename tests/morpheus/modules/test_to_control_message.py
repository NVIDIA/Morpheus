#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import pytest

# When segment modules are imported, they're added to the module registry.
# To avoid flake8 warnings about unused code, the noqa flag is used during import.
import morpheus.loaders  # noqa: F401 # pylint: disable=unused-import
import morpheus.modules  # noqa: F401 # pylint: disable=unused-import
# pylint: enable=unused-import
from morpheus.pipeline.pipeline import Pipeline
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_ids import TO_CONTROL_MESSAGE


def test_contains_namespace():
    registry = mrc.ModuleRegistry

    assert registry.contains_namespace("morpheus")


def test_is_version_compatible():
    registry = mrc.ModuleRegistry

    release_version = [int(x) for x in mrc.__version__.split(".")]
    old_release_version = [22, 10, 0]
    no_version_patch = [22, 10]
    no_version_minor_and_patch = [22]

    assert registry.is_version_compatible(release_version)
    assert registry.is_version_compatible(old_release_version) is not True
    assert registry.is_version_compatible(no_version_patch) is not True
    assert registry.is_version_compatible(no_version_minor_and_patch) is not True


def test_get_module():
    registry = mrc.ModuleRegistry

    fn_constructor = registry.get_module_constructor("ToControlMessage", "morpheus")
    assert fn_constructor is not None

    config = {}
    module_instance = fn_constructor("ToControlMessageTest", config)  # pylint: disable=not-callable
    assert isinstance(module_instance, mrc.core.segment.SegmentModule)


@pytest.mark.gpu_mode
@pytest.mark.parametrize("expected_count", [1, 2])
def test_to_control_message_module(config, filter_probs_df, expected_count):
    dataframes = [filter_probs_df for _ in range(expected_count)]

    pipe = Pipeline(config)

    # Create the stages
    source_stage = pipe.add_stage(InMemorySourceStage(config, dataframes))

    to_cm_module_config = {
        "module_id": TO_CONTROL_MESSAGE,
        "module_name": "to_control_message",
        "namespace": MORPHEUS_MODULE_NAMESPACE,
        "meta_data": {
            "data_type": "streaming"
        },
        "tasks": [{
            "type": "inference", "properties": {}
        }]
    }

    to_control_message_stage = pipe.add_stage(
        LinearModulesStage(config, to_cm_module_config, input_port_name="input", output_port_name="output"))

    sink_stage = pipe.add_stage(InMemorySinkStage(config))

    pipe.add_edge(source_stage, to_control_message_stage)
    pipe.add_edge(to_control_message_stage, sink_stage)

    pipe.run()

    assert len(sink_stage.get_messages()) == expected_count
