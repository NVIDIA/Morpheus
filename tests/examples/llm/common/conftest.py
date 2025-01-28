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

import os
import sys
import types

import pytest

from _utils import TEST_DIRS


@pytest.fixture(scope="function")
def import_utils(restore_sys_path):  # pylint: disable=unused-argument
    utils_path = os.path.join(TEST_DIRS.examples_dir, 'llm/common/')
    sys.path.insert(0, utils_path)

    import utils

    return utils


@pytest.fixture(scope="function")
def import_web_scraper_module(restore_sys_path):  # pylint: disable=unused-argument
    web_scraper_path = os.path.join(TEST_DIRS.examples_dir, 'llm/vdb_upload/module')
    sys.path.insert(0, web_scraper_path)

    import web_scraper_module

    return web_scraper_module


# Fixture for importing the module
@pytest.fixture(scope="function")
def import_content_extractor_module(restore_sys_path, pypdfium2: types.ModuleType):  # pylint: disable=unused-argument
    sys.path.insert(0, os.path.join(TEST_DIRS.examples_dir, 'llm/vdb_upload/module/'))

    import content_extractor_module

    return content_extractor_module


@pytest.fixture(name="langchain", autouse=True, scope='session')
def langchain_fixture(langchain):
    """
    All the tests in this subdir require langchain
    """
    yield langchain
