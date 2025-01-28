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

# Disable redefined-outer-name, it doesn't detect fixture name usage correctly and reports errors that are not errors.
# pylint: disable=redefined-outer-name

import os
import shutil
import tempfile

import pandas as pd
import pytest

from morpheus.io.data_storage import FileSystemStorage
from morpheus.io.data_storage.file_system import row_count_from_file


# Fixtures
@pytest.fixture
def csv_data():
    return pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})


@pytest.fixture
def parquet_data():
    return pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})


@pytest.fixture
def csv_file(csv_data):
    temp_dir = tempfile.mkdtemp()

    filepath = f'{temp_dir}/test_data.csv'
    csv_data.to_csv(filepath, index=False)
    yield filepath

    shutil.rmtree(temp_dir)


@pytest.fixture
def parquet_file(parquet_data):
    temp_dir = tempfile.mkdtemp()

    filepath = f'{temp_dir}/test_data.parquet'
    parquet_data.to_parquet(filepath, index=False)
    yield filepath

    shutil.rmtree(temp_dir)


# Tests
def test_csv(csv_data):
    storage = FileSystemStorage('test_data.csv', 'csv')
    storage.store(csv_data)

    assert storage.backing_source == 'test_data.csv'
    assert storage.num_rows == 3
    assert storage.owner is True

    loaded_data = storage.load()
    pd.testing.assert_frame_equal(loaded_data, csv_data)

    storage.delete()
    assert not os.path.exists('test_data.csv')


def test_parquet(parquet_data):
    storage = FileSystemStorage(file_path="test_data.parquet", file_format='parquet')
    storage.store(parquet_data)

    assert storage.backing_source == 'test_data.parquet'
    assert storage.num_rows == 3
    assert storage.owner is True

    loaded_data = storage.load()
    pd.testing.assert_frame_equal(loaded_data, parquet_data)

    storage.delete()
    assert not os.path.exists('test_data.parquet')


def test_copy_from_source(csv_file, parquet_file):
    # Test store from file and load with copy_from_source=True
    storage_csv = FileSystemStorage('test_data_copy.csv', 'csv')
    storage_csv.store(csv_file, copy_from_source=True)
    assert storage_csv.backing_source == 'test_data_copy.csv'
    assert storage_csv.num_rows == 3
    assert storage_csv.owner is True
    assert os.path.exists('test_data_copy.csv')

    storage_parquet = FileSystemStorage('test_data_copy.parquet', 'parquet')
    storage_parquet.store(parquet_file, copy_from_source=True)
    assert storage_parquet.backing_source == 'test_data_copy.parquet'
    assert storage_parquet.num_rows == 3
    assert storage_parquet.owner is True
    assert os.path.exists('test_data_copy.parquet')

    # Clean up
    storage_csv.delete()
    storage_parquet.delete()


def test_no_copy_from_source(csv_file, parquet_file):
    # Test store from file and load with copy_from_source=False
    storage_csv = FileSystemStorage('test_data_link.csv', 'csv')
    storage_csv.store(csv_file, copy_from_source=False)
    assert storage_csv.backing_source == csv_file
    assert storage_csv.num_rows == 3
    assert storage_csv.owner is False

    storage_parquet = FileSystemStorage('test_data_link.parquet', 'parquet')
    storage_parquet.store(parquet_file, copy_from_source=False)
    assert storage_parquet.backing_source == parquet_file
    assert storage_parquet.num_rows == 3
    assert storage_parquet.owner is False


def test_invalid_file_format():
    # Test that an error is raised when an unsupported file format is used
    with pytest.raises(NotImplementedError):
        FileSystemStorage('test_data', 'txt')


def test_delete_non_owner(csv_file, parquet_file):
    # Test that delete does not delete the file if it is not the owner
    storage_csv = FileSystemStorage('test_data_link.csv', 'csv')
    storage_csv.store(csv_file, copy_from_source=False)
    storage_csv.delete()
    assert os.path.exists(csv_file)

    storage_parquet = FileSystemStorage('test_data_link.csv', 'parquet')
    storage_parquet.store(parquet_file, copy_from_source=False)
    storage_parquet.delete()
    assert os.path.exists(parquet_file)


def test_row_count_from_file(csv_file, parquet_file):
    # Test that the row count is correctly computed from a file
    assert row_count_from_file(csv_file, 'csv') == 3
    assert row_count_from_file(parquet_file, 'parquet') == 3


def test_row_count_from_file_no_hint(csv_file, parquet_file):
    # Test that the row count is correctly computed from a file without a file format hint
    assert row_count_from_file(csv_file) == 3
    assert row_count_from_file(parquet_file) == 3


def test_row_count_from_file_invalid_file_format():
    # Test that an error is raised when an unsupported file format is used
    with pytest.raises(ValueError):
        row_count_from_file('test_data.txt')
