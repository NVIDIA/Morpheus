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

import glob
import os
from unittest import mock

import pytest

from _utils import TEST_DIRS
from morpheus.config import Config
from morpheus.pipeline.single_output_source import SingleOutputSource


def test_constructor(config: Config):
    from morpheus_dfp.stages.multi_file_source import MultiFileSource

    batch_size = 1234
    n_threads = 13
    config.pipeline_batch_size = batch_size
    config.num_threads = n_threads
    filenames = ['some/file', '/tmp/some/files-2023-*-*.csv', 's3://some/bucket/2023-*-*.csv.gz']
    stage = MultiFileSource(config, filenames=filenames, watch=False, watch_interval=2.1)

    assert isinstance(stage, SingleOutputSource)
    assert stage._batch_size == batch_size
    assert stage._max_concurrent == n_threads
    assert stage._filenames == filenames
    assert not stage._watch
    assert stage._watch_interval == 2.1


def test_generate_frames_fsspec(config: Config, tmp_path: str):
    from morpheus_dfp.stages.multi_file_source import MultiFileSource

    file_glob = os.path.join(TEST_DIRS.tests_data_dir, 'appshield', 'snapshot-1', '*.json')
    temp_glob = os.path.join(tmp_path, '*.json')  # this won't match anything
    stage = MultiFileSource(config, filenames=[file_glob, temp_glob], watch=False)

    fsspec_gen = stage._generate_frames_fsspec()
    specs = next(fsspec_gen)

    files = sorted(f.path for f in specs)
    assert files == sorted(glob.glob(file_glob))

    # Verify that we are not in watch mode
    with open(os.path.join(tmp_path, 'new-file.json'), 'w', encoding='utf-8') as f:
        f.write('{"foo": "bar"}')

    with pytest.raises(StopIteration):
        next(fsspec_gen)


@mock.patch('time.sleep')
def test_polling_generate_frames_fsspec(amock_time: mock.MagicMock, config: Config, tmp_path: str):
    from morpheus_dfp.stages.multi_file_source import MultiFileSource

    file_glob = os.path.join(TEST_DIRS.tests_data_dir, 'appshield', 'snapshot-1', '*.json')
    temp_glob = os.path.join(tmp_path, '*.json')  # this won't match anything
    stage = MultiFileSource(config, filenames=[file_glob, temp_glob], watch=True, watch_interval=0.2)

    fsspec_gen = stage._polling_generate_frames_fsspec()
    specs = next(fsspec_gen)

    files = sorted(f.path for f in specs)
    assert files == sorted(glob.glob(file_glob))

    # Verify that we are not in watch mode
    with open(os.path.join(tmp_path, 'new-file.json'), 'w', encoding='utf-8') as f:
        f.write('{"foo": "bar"}')

    specs = next(fsspec_gen)
    assert len(specs) == 1
    assert specs[0].path == os.path.join(tmp_path, 'new-file.json')
    amock_time.assert_called_once()


def test_generate_frames_fsspec_no_files(config: Config, tmp_path: str):
    from morpheus_dfp.stages.multi_file_source import MultiFileSource

    assert os.listdir(tmp_path) == []

    filenames = [os.path.join(tmp_path, '*.csv')]
    stage = MultiFileSource(config, filenames=filenames, watch=False)

    with pytest.raises(RuntimeError):
        next(stage._generate_frames_fsspec())
