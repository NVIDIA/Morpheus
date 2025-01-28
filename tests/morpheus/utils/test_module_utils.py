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

from unittest import mock

import mrc
import pytest

from morpheus.utils.module_utils import load_module
from morpheus.utils.module_utils import make_nested_module
from morpheus.utils.module_utils import mrc_version
from morpheus.utils.module_utils import register_module
from morpheus.utils.module_utils import verify_module_registration

# pylint: disable=unused-argument,too-many-function-args


def test_mrc_version():
    assert len(mrc_version) == 3
    assert isinstance(mrc_version, list)
    assert isinstance(mrc_version[0], int)
    assert isinstance(mrc_version[1], int)
    assert isinstance(mrc_version[2], int)


def test_register_module():

    @register_module("TestModule", "test_morpheus_modules")
    def module_init(builder: mrc.Builder):
        return True

    assert module_init(None)

    # Attempting to register duplicate module raises an error.
    with pytest.raises(TypeError):

        @register_module(None, "test_morpheus_modules")
        def module_init2(builder: mrc.Builder):
            pass

        module_init2(None)


def test_verify_module_registration():
    module_config = {"module_id": "TestModule", "namespace": "test_morpheus_modules"}
    module_config2 = {"module_id": "TestModule", "namespace": "test_morpheus_modules", "module_name": "test_module"}

    @verify_module_registration
    def verify_module_existence(config):
        return True

    assert verify_module_existence(module_config2)

    with pytest.raises(KeyError):
        verify_module_existence(module_config)

    with pytest.raises(TypeError):
        verify_module_existence(None, "test_morpheus_modules")

    with pytest.raises(Exception):
        verify_module_existence("TestModule2", "test_morpheus_modules")

    with pytest.raises(Exception):
        verify_module_existence("TestModule", "default")


def test_load_module():
    mock_builder = mock.MagicMock()
    mock_module = mock.MagicMock()

    mock_builder.load_module.return_value = mock_module

    config = {"module_id": "TestModule", "namespace": "test_morpheus_modules", "module_name": "test_module"}
    module = load_module(config, builder=mock_builder)

    assert module is not None
    mock_builder.load_module.assert_called_once()


def test_make_nested_module():
    registry = mrc.ModuleRegistry

    conf_module1 = {"module_id": "InnerModule1", "namespace": "test_morpheus_modules", "module_name": "inner_module1"}
    conf_module2 = {"module_id": "InnerModule2", "namespace": "test_morpheus_modules", "module_name": "inner_module2"}
    conf_module3 = {"module_id": "InnerModule3", "namespace": "test_morpheus_modules", "module_name": "inner_module3"}

    @register_module("InnerModule1", "test_morpheus_modules")
    def module_init1(builder: mrc.Builder):
        pass

    @register_module("InnerModule2", "test_morpheus_modules")
    def module_init2(builder: mrc.Builder):
        pass

    @register_module("InnerModule3", "test_morpheus_modules")
    def module_init3(builder: mrc.Builder):
        pass

    ordered_inner_modules_meta = [conf_module1, conf_module2, conf_module3]

    make_nested_module("ModuleExpose", "test_morpheus_modules", ordered_inner_modules_meta)

    assert registry.contains("ModuleExpose", "test_morpheus_modules")
