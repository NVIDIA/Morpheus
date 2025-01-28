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

import cudf

from morpheus.io.data_record import DataRecord


@pytest.fixture(scope='session')
def test_data():
    # setup part
    temp_dir = tempfile.mkdtemp()
    test_cudf_dataframe = cudf.DataFrame({'a': [9, 10], 'b': [11, 12]})
    test_pd_dataframe = pd.DataFrame({'a': [13, 14], 'b': [15, 16]})
    test_parquet_filepath = os.path.join(temp_dir, 'test_file.parquet')
    test_csv_filepath = os.path.join(temp_dir, 'test_file.csv')

    test_cudf_dataframe.to_parquet(test_parquet_filepath)
    test_cudf_dataframe.to_csv(test_csv_filepath, index=False, header=True)

    yield {
        'cudf_dataframe': test_cudf_dataframe,
        'pd_dataframe': test_pd_dataframe,
        'parquet_filepath': test_parquet_filepath,
        'csv_filepath': test_csv_filepath
    }

    # teardown part
    shutil.rmtree(temp_dir)


@pytest.mark.parametrize("storage_type", ['in_memory', 'filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_load(storage_type, file_format, test_data):
    data_record = DataRecord(data_source=test_data['cudf_dataframe'],
                             data_label='test_data',
                             storage_type=storage_type,
                             file_format=file_format)
    loaded_df = data_record.load()
    pd.testing.assert_frame_equal(loaded_df, test_data['cudf_dataframe'].to_pandas())


@pytest.mark.parametrize("storage_type", ['in_memory', 'filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_num_rows(storage_type, file_format, test_data):
    data_record = DataRecord(data_source=test_data['cudf_dataframe'],
                             data_label='test_data',
                             storage_type=storage_type,
                             file_format=file_format)
    num_rows = data_record.num_rows
    assert num_rows == len(test_data['cudf_dataframe'])


@pytest.mark.parametrize("storage_type", ['invalid', "something else invalid"])
def test_invalid_storage_type(storage_type, test_data):
    with pytest.raises(ValueError):
        DataRecord(data_source=test_data['cudf_dataframe'],
                   data_label='test_data',
                   storage_type=storage_type,
                   file_format='parquet')


@pytest.mark.parametrize("file_format", ['invalid', "something else invalid"])
def test_invalid_data_format(file_format, test_data):
    with pytest.raises(NotImplementedError):
        DataRecord(data_source=test_data['cudf_dataframe'],
                   data_label='test_data',
                   storage_type='in_memory',
                   file_format=file_format)


def test_deletion_filesystem_csv():
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'temp_data.csv')
    df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df.to_csv(temp_file)

    data_record = DataRecord(temp_file, 'test_label', 'filesystem', 'csv', copy_from_source=True)
    path_on_disk = data_record.backing_source

    del data_record
    assert not os.path.exists(path_on_disk)

    shutil.rmtree(temp_dir)


def test_deletion_filesystem_parquet():
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'temp_data.parquet')
    df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df.to_parquet(temp_file)

    data_record = DataRecord(temp_file, 'test_label', 'filesystem', 'parquet', copy_from_source=True)
    path_on_disk = data_record.backing_source

    del data_record
    assert not os.path.exists(path_on_disk)

    shutil.rmtree(temp_dir)


def test_deletion_no_owner():
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'temp_data.csv')
    df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df.to_csv(temp_file)

    data_record = DataRecord(temp_file, 'test_label', 'filesystem', 'csv', copy_from_source=False)
    path_on_disk = data_record.backing_source

    del data_record
    assert os.path.exists(path_on_disk)

    shutil.rmtree(temp_dir)


def test_properties_csv():
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'temp_data.csv')
    df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df.to_csv(temp_file, index=False, header=True)

    data_record = DataRecord(temp_file, f'{temp_dir}/test_label.csv', 'filesystem', 'csv', copy_from_source=True)

    assert data_record.backing_source == f'{temp_dir}/test_label.csv'
    assert data_record.format == 'csv'
    assert data_record.num_rows == 3

    pd.testing.assert_frame_equal(data_record.data, df.to_pandas())

    del data_record
    shutil.rmtree(temp_dir)


def test_properties_parquet():
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'temp_data.parquet')
    df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df.to_parquet(temp_file)

    data_record = DataRecord(temp_file, 'test_label', 'filesystem', 'parquet', copy_from_source=True)

    assert data_record.data_label == 'test_label'
    pd.testing.assert_frame_equal(data_record.data, df.to_pandas())
    assert data_record.format == 'parquet'
    assert data_record.num_rows == 3

    del data_record
    shutil.rmtree(temp_dir)


def test_properties_in_memory():
    df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    data_record = DataRecord(df, 'test_label', 'in_memory', 'csv')

    assert data_record.backing_source == 'IO Buffer'
    pd.testing.assert_frame_equal(data_record.data, df.to_pandas())
    assert data_record.format == 'csv'
    assert data_record.num_rows == 3

    del data_record
