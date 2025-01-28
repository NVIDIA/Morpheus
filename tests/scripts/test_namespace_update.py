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

import importlib.util
import os
import types

from _utils import TEST_DIRS


def copy_data_to_tmp_path(tmp_path) -> str:
    '''
    Copy the data to a temporary directory as we will be modifying the files.
    '''
    data_dir = os.path.join(TEST_DIRS.tests_dir, "scripts/data")
    tmp_data_dir = tmp_path / "scripts"
    tmp_data_dir.mkdir()
    os.system(f"cp -r {data_dir} {tmp_data_dir}")
    scripts_data_dir = os.path.join(tmp_data_dir, "data")
    return scripts_data_dir


def import_module_from_path(module_name, path) -> tuple:
    '''
    Import a module from the pytest tmp_path.
    '''
    # Create a module spec from the given path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if not spec:
        return None, None

    # Load the module from the created spec
    module = importlib.util.module_from_spec(spec)
    if not module:
        return None, None

    return spec, module


def test_dfp_namespace_update(tmp_path):
    '''
    Update the DFP namespace imports and verify the imports work.
    '''
    scripts_data_dir = copy_data_to_tmp_path(tmp_path)
    module_name = 'dfp_old_namespace_data'
    module_path = os.path.join(scripts_data_dir, f'{module_name}.py')

    # check if the file exists in the tmp_path
    assert os.path.exists(module_path), f"Failed to copy {module_name} to {scripts_data_dir}"

    # dfp imports expected to fail before namespace update
    spec, module = import_module_from_path(module_name, module_path)
    assert module is not None, f"Failed to import {module_name} from {module_path}"
    try:
        spec.loader.exec_module(module)
        assert False, "dfp_namespace_data input is not setup with the old imports"
    except ModuleNotFoundError:
        pass

    # update imports to the new namespace by running morpheus_namespace_update.py
    update_namespace_script = os.path.join(TEST_DIRS.morpheus_root, "scripts/morpheus_namespace_update.py")
    os.system(f"python {update_namespace_script} --directory {scripts_data_dir} --dfp")

    # verify the morpheus_dfp imports work
    spec, module = import_module_from_path(module_name, module_path)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError:
        assert False, "old dfp imports are not updated to the new namespace"


def test_llm_namespace_update(tmp_path, pymilvus: types.ModuleType):  # pylint: disable=unused-argument
    '''
    Update the LLM namespace imports and verify the imports work.
    '''
    scripts_data_dir = copy_data_to_tmp_path(tmp_path)
    module_name = 'llm_old_namespace_data'
    module_path = os.path.join(scripts_data_dir, f'{module_name}.py')

    # check if the file exists in the tmp_path
    assert os.path.exists(module_path), f"Failed to copy {module_name} to {scripts_data_dir}"

    # llm imports expected to fail before namespace update
    spec, module = import_module_from_path(module_name, module_path)
    assert module is not None, f"Failed to import {module_name} from {module_path}"
    try:
        spec.loader.exec_module(module)
        assert False, "llm_namespace_data input is not setup with the old imports"
    except ModuleNotFoundError:
        pass

    # update imports to the new namespace by running morpheus_namespace_update.py
    update_namespace_script = os.path.join(TEST_DIRS.morpheus_root, "scripts/morpheus_namespace_update.py")
    os.system(f"python {update_namespace_script} --directory {scripts_data_dir} --llm")

    # verify the morpheus_llm imports work
    spec, module = import_module_from_path(module_name, module_path)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError:
        assert False, "old llm imports are not updated to the new namespace"
