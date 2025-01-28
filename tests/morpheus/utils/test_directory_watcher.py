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

import pytest

from _utils import TEST_DIRS
from morpheus.utils.directory_watcher import DirectoryWatcher


@pytest.mark.parametrize('watch_directory', [True])
@pytest.mark.parametrize('max_files', [-1])
@pytest.mark.parametrize('sort_glob', [True])
@pytest.mark.parametrize('recursive', [True])
@pytest.mark.parametrize('queue_max_size', [128])
@pytest.mark.parametrize('batch_timeout', [5.0])
def test_constructor(watch_directory, max_files, sort_glob, recursive, queue_max_size, batch_timeout):
    input_glob = os.path.join(TEST_DIRS.tests_data_dir, 'appshield', '*', '*.json')
    watcher = DirectoryWatcher(input_glob,
                               watch_directory,
                               max_files,
                               sort_glob,
                               recursive,
                               queue_max_size,
                               batch_timeout)

    assert watcher._sort_glob
    assert watcher._watch_directory
    assert watcher._max_files == -1
