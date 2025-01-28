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

import typing
from unittest import mock

import mrc
import pytest

from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from morpheus.utils.module_utils import mrc_version


@pytest.fixture(name="module_config")
def module_config_fixture():
    return {"module_id": "TestSimpleModule", "module_name": "test_simple_module", "namespace": "test_morpheus_modules"}


@pytest.mark.gpu_and_cpu_mode
def test_constructor(config, module_config: dict):

    mod_stage = LinearModulesStage(config, module_config, input_port_name="test_in", output_port_name="test_out")

    assert mod_stage.name == "test_simple_module"

    # Just ensure that we get a valid non-empty tuple
    accepted_types = mod_stage.accepted_types()
    assert isinstance(accepted_types, tuple)
    assert len(accepted_types) > 0
    assert accepted_types[0] == typing.Any

    pytest.raises(NotImplementedError, mod_stage._get_cpp_module_node, None)


@pytest.mark.gpu_and_cpu_mode
def test_build_single_before_module_registration(config, module_config: dict):

    mock_node = mock.MagicMock()
    mock_segment = mock.MagicMock()
    mock_module = mock.MagicMock()
    mock_input_stream = mock.MagicMock()

    mock_segment.load_module.return_value = mock_module
    mock_segment.make_node.return_value = mock_node

    mod_stage = LinearModulesStage(config, module_config, input_port_name="test_in", output_port_name="test_out")

    with pytest.raises(Exception):
        mod_stage._build_single(mock_segment, mock_input_stream)


def register_test_module(id_postfix: str):
    registry = mrc.ModuleRegistry

    def module_init_fn(_: mrc.Builder):
        pass

    registry.register_module(f"TestSimpleModule_{id_postfix}", "test_morpheus_modules", mrc_version, module_init_fn)


@pytest.mark.gpu_and_cpu_mode
def test_build_single_after_module_registration(config, module_config: dict):

    register_test_module(config.execution_mode.value)
    module_config["module_id"] = f"{module_config['module_id']}_{config.execution_mode.value}"

    mock_node = mock.MagicMock()
    mock_segment = mock.MagicMock()
    mock_module = mock.MagicMock()
    mock_input_stream = mock.MagicMock()

    mock_segment.load_module.return_value = mock_module
    mock_segment.make_node.return_value = mock_node

    mod_stage = LinearModulesStage(config, module_config, input_port_name="test_in", output_port_name="test_out")

    mod_stage._build_single(mock_segment, mock_input_stream)

    mock_segment.load_module.assert_called_once()
    mock_segment.make_edge.assert_called_once()
