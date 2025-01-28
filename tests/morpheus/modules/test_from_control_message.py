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

import os

import mrc
import pytest

# When segment modules are imported, they're added to the module registry.
# To avoid flake8 & pylint warnings about unused code, the noqa flag and pylint directive is used during import.
import morpheus.loaders  # noqa: F401 # pylint: disable=unused-import
import morpheus.modules  # noqa: F401 # pylint: disable=unused-import
from _utils import TEST_DIRS
from morpheus.pipeline.pipeline import Pipeline
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from morpheus.stages.input.control_message_file_source_stage import ControlMessageFileSourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils.loader_ids import FSSPEC_LOADER
from morpheus.utils.module_ids import DATA_LOADER
from morpheus.utils.module_ids import FROM_CONTROL_MESSAGE
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE


@pytest.fixture(name="filename", scope="function")
def filename_fixture(request):
    test_data_dir = os.path.join(TEST_DIRS.tests_data_dir, "control_messages")
    f_name = request.param
    yield os.path.join(test_data_dir, f_name)


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

    fn_constructor = registry.get_module_constructor("FromControlMessage", "morpheus")
    assert fn_constructor is not None

    config = {}
    fn_constructor("FromControlMessageTest", config)  # pylint: disable=not-callable


@pytest.mark.gpu_mode
@pytest.mark.parametrize("filename, expected_count", [("train_infer.json", 0), ("train.json", 0)],
                         indirect=["filename"])
def test_cm_with_no_payload(config, filename, expected_count):
    from_cm_module_config = {
        "module_id": FROM_CONTROL_MESSAGE,
        "module_name": "from_control_message",
        "namespace": MORPHEUS_MODULE_NAMESPACE
    }

    pipe = Pipeline(config)

    # Create the stages
    source_stage = pipe.add_stage(ControlMessageFileSourceStage(config, filenames=[filename]))
    from_control_message_stage = pipe.add_stage(
        LinearModulesStage(config, from_cm_module_config, input_port_name="input", output_port_name="output"))
    sink_stage = pipe.add_stage(InMemorySinkStage(config))

    pipe.add_edge(source_stage, from_control_message_stage)
    pipe.add_edge(from_control_message_stage, sink_stage)

    pipe.run()

    assert len(sink_stage.get_messages()) == expected_count


@pytest.mark.gpu_mode
@pytest.mark.parametrize("filename, expected_count", [("train_infer.json", 2), ("train.json", 1)],
                         indirect=["filename"])
def test_cm_with_with_payload(config, filename, expected_count):
    from_cm_module_config = {
        "module_id": FROM_CONTROL_MESSAGE,
        "module_name": "from_control_message",
        "namespace": MORPHEUS_MODULE_NAMESPACE
    }

    fsspec_dataloader_module_config = {
        "module_id": DATA_LOADER,
        "module_name": "fsspec_dataloader",
        "namespace": MORPHEUS_MODULE_NAMESPACE,
        "loaders": [{
            "id": FSSPEC_LOADER
        }]
    }

    pipe = Pipeline(config)

    # Create the stages
    source_stage = pipe.add_stage(ControlMessageFileSourceStage(config, filenames=[filename]))
    fsspec_dataloader_stage = pipe.add_stage(
        LinearModulesStage(config, fsspec_dataloader_module_config, input_port_name="input", output_port_name="output"))
    from_control_message_stage = pipe.add_stage(
        LinearModulesStage(config, from_cm_module_config, input_port_name="input", output_port_name="output"))
    sink_stage = pipe.add_stage(InMemorySinkStage(config))

    pipe.add_edge(source_stage, fsspec_dataloader_stage)
    pipe.add_edge(fsspec_dataloader_stage, from_control_message_stage)
    pipe.add_edge(from_control_message_stage, sink_stage)

    pipe.run()

    # We should expect to see three messages in the sink as three sources have generated one message each.
    assert len(sink_stage.get_messages()) == expected_count
