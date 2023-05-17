# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import functools
import os
import re
from datetime import datetime
from datetime import timezone

import fsspec
import pytest

from morpheus.config import Config
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.file_utils import date_extractor
from utils import TEST_DIRS


def test_constructor(config: Config):
    from dfp.stages.dfp_file_batcher_stage import DFPFileBatcherStage

    date_conversion_func = lambda x: x  # noqa E731
    stage = DFPFileBatcherStage(config,
                                date_conversion_func,
                                'M',
                                sampling=55,
                                start_time=datetime(1999, 1, 1),
                                end_time=datetime(2005, 10, 11, 4, 34, 21))

    assert isinstance(stage, SinglePortStage)
    assert stage._date_conversion_func is date_conversion_func
    assert stage._sampling == 55
    assert stage._period == 'M'
    assert stage._start_time == datetime(1999, 1, 1)
    assert stage._end_time == datetime(2005, 10, 11, 4, 34, 21)


def test_constructor_deprecated_args(config: Config):
    """Test that the deprecated sampling_rate_s arg is still supported"""

    from dfp.stages.dfp_file_batcher_stage import DFPFileBatcherStage

    with pytest.deprecated_call():
        stage = DFPFileBatcherStage(config, lambda x: x, sampling_rate_s=55)

    assert isinstance(stage, SinglePortStage)
    assert stage._sampling == "55S"


def test_constructor_both_sample_args_error(config: Config):
    """Test that an error is raised if both sampling and sampling_rate_s are specified"""
    from dfp.stages.dfp_file_batcher_stage import DFPFileBatcherStage

    with pytest.raises(AssertionError):
        DFPFileBatcherStage(config, lambda x: x, sampling=55, sampling_rate_s=20)


def test_on_data(config: Config):
    from dfp.stages.dfp_file_batcher_stage import DFPFileBatcherStage

    date_re = re.compile(r"(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})"
                         r"_(?P<hour>\d{1,2})-(?P<minute>\d{1,2})-(?P<second>\d{1,2})(?P<microsecond>\.\d{1,6})")

    date_conversion_func = functools.partial(date_extractor, filename_regex=date_re)

    stage = DFPFileBatcherStage(config, date_conversion_func)

    assert stage.on_data([]) == []

    test_data_dir = os.path.join(TEST_DIRS.tests_data_dir, 'appshield', 'snapshot-1')
    file_specs = fsspec.open_files(os.path.join(test_data_dir, '*.json'))

    # With a one-day batch all files will fit in the batch
    batches = stage.on_data(file_specs)
    assert len(batches) == 1

    batch = batches[0]
    assert sorted(f.path for f in batch[0]) == sorted(f.path for f in file_specs)
    assert batch[1] == 1

    # Test with a one-minute window which should split the data into two batches
    stage = DFPFileBatcherStage(config, date_conversion_func, period='min')
    batches = stage.on_data(file_specs)
    assert len(batches) == 2

    expected_10_25_files = sorted(f.path
                                  for f in fsspec.open_files(os.path.join(test_data_dir, '*_2022-01-30_10-25*.json')))
    expected_10_26_files = sorted(f.path
                                  for f in fsspec.open_files(os.path.join(test_data_dir, '*_2022-01-30_10-26*.json')))

    (b1, b2) = batches
    assert sorted(f.path for f in b1[0]) == expected_10_25_files
    assert b1[1] == 2

    assert sorted(f.path for f in b2[0]) == expected_10_26_files
    assert b2[1] == 2

    # Test with a start time that excludes some files
    stage = DFPFileBatcherStage(config,
                                date_conversion_func,
                                period='min',
                                start_time=datetime(2022, 1, 30, 10, 26, tzinfo=timezone.utc))
    batches = stage.on_data(file_specs)
    assert len(batches) == 1

    batch = batches[0]
    assert sorted(f.path for f in batch[0]) == expected_10_26_files
    assert batch[1] == 1

    # Test with a end time that excludes some files
    stage = DFPFileBatcherStage(config,
                                date_conversion_func,
                                period='min',
                                end_time=datetime(2022, 1, 30, 10, 26, tzinfo=timezone.utc))
    batches = stage.on_data(file_specs)
    assert len(batches) == 1

    batch = batches[0]
    assert sorted(f.path for f in batch[0]) == expected_10_25_files
    assert batch[1] == 1

    # Test with a start & end time that excludes some files
    stage = DFPFileBatcherStage(config,
                                date_conversion_func,
                                period='min',
                                start_time=datetime(2022, 1, 30, 10, 26, 0, tzinfo=timezone.utc),
                                end_time=datetime(2022, 1, 30, 10, 26, 3, tzinfo=timezone.utc))
    batches = stage.on_data(file_specs)
    assert len(batches) == 1

    expected_files = [f for f in expected_10_26_files if not f.endswith('04.570268.json')]

    batch = batches[0]
    assert sorted(f.path for f in batch[0]) == expected_files
    assert batch[1] == 1
