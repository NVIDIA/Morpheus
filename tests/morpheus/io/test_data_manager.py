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
import unittest
import uuid

import pandas as pd
import pytest

import cudf

from morpheus.io import data_manager

DataManager = data_manager.DataManager

sources = [
    cudf.DataFrame({
        'a': [1, 2], 'b': [3, 4]
    }),
    'buffer2.parquet',  # Local or remote file path
    pd.DataFrame({
        'a': [5, 6], 'b': [7, 8]
    }),
]


@pytest.fixture(scope='session')
def dataframe_fixture_data():
    # Create temporary file paths
    temp_dir = tempfile.mkdtemp()

    parquet_filepath = f"{temp_dir}/test_file.parquet"
    csv_filepath = f"{temp_dir}/test_file.csv"

    # Create test data
    test_cudf_dataframe = cudf.DataFrame({'a': [9, 10], 'b': [11, 12]})
    test_pd_dataframe = pd.DataFrame({'a': [13, 14], 'b': [15, 16]})

    # Write data to temporary files
    test_cudf_dataframe.to_parquet(parquet_filepath)
    test_cudf_dataframe.to_csv(csv_filepath, index=False, header=True)

    # Provide the file paths and data as a dictionary
    yield {
        'test_cudf_dataframe': test_cudf_dataframe,
        'test_pd_dataframe': test_pd_dataframe,
        'test_parquet_filepath': parquet_filepath,
        'test_csv_filepath': csv_filepath
    }

    shutil.rmtree(temp_dir)


@pytest.mark.parametrize("storage_type", ['in_memory'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_memory_storage(storage_type, file_format, dataframe_fixture_data):
    data = dataframe_fixture_data["test_cudf_dataframe"]

    data_manager = DataManager(storage_type=storage_type, file_format=file_format)
    assert len(data_manager) == 0

    sid = data_manager.store(data)
    assert len(data_manager) == 1
    assert sid in data_manager


@pytest.mark.parametrize("storage_type", ['in_memory', 'filesystem'])
def test_filesystem_storage_type(storage_type):
    data_manager = DataManager(storage_type=storage_type)
    assert (len(data_manager) == 0)
    assert (data_manager.storage_type == storage_type)


@pytest.mark.parametrize("storage_type", ['invalid', "something else invalid"])
def test_invalid_storage_type(storage_type):
    with pytest.raises(ValueError):
        data_manager = DataManager(storage_type=storage_type)  # noqa pylint: disable=unused-variable


@pytest.mark.parametrize("storage_type", ['in_memory', 'filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_add_remove_source(storage_type, file_format):
    data_manager = DataManager(storage_type=storage_type, file_format=file_format)
    new_source = pd.DataFrame({'a': [9, 10], 'b': [11, 12]})

    sid = data_manager.store(new_source)
    assert (len(data_manager) == 1)
    data_manager.remove(sid)
    assert (len(data_manager) == 0)


@pytest.mark.parametrize("storage_type", ['filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_filesystem_storage_files_exist(storage_type, file_format, dataframe_fixture_data):
    test_cudf_dataframe = dataframe_fixture_data["test_cudf_dataframe"]
    test_pd_dataframe = dataframe_fixture_data["test_pd_dataframe"]

    data_manager = DataManager(storage_type=storage_type, file_format=file_format)
    sid1 = data_manager.store(test_cudf_dataframe)
    sid2 = data_manager.store(test_pd_dataframe)

    files = data_manager.manifest
    for file_path in files.values():
        assert (os.path.exists(file_path))

    data_manager.remove(sid1)
    data_manager.remove(sid2)

    files = data_manager.manifest
    for file_path in files:
        assert (not os.path.exists(file_path))


@pytest.mark.parametrize("storage_type", ['filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_large_fileset_filesystem_storage(storage_type, file_format):
    num_dataframes = 100
    dataframes = [cudf.DataFrame({'a': [i, i + 1], 'b': [i + 2, i + 3]}) for i in range(num_dataframes)]
    data_manager = DataManager(storage_type=storage_type, file_format=file_format)

    source_ids = [data_manager.store(df) for df in dataframes]
    assert (len(data_manager) == num_dataframes)

    for source_id in source_ids:
        assert (source_id in data_manager)

    files = data_manager.manifest.values()
    for file_path in files:
        assert (os.path.exists(file_path))

    for source_id in source_ids:
        data_manager.remove(source_id)

    assert (len(data_manager) == 0)

    for file_path in files:
        assert (not os.path.exists(file_path))


@pytest.mark.parametrize("storage_type", ['in_memory', 'filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_load_cudf_dataframe(storage_type, file_format, dataframe_fixture_data):
    test_cudf_dataframe = dataframe_fixture_data["test_cudf_dataframe"]

    data_manager = DataManager(storage_type=storage_type, file_format=file_format)
    sid = data_manager.store(test_cudf_dataframe)
    loaded_df = data_manager.load(sid)

    pd.testing.assert_frame_equal(loaded_df, test_cudf_dataframe.to_pandas())


@pytest.mark.parametrize("storage_type", ['in_memory', 'filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_load_pd_dataframe(storage_type, file_format, dataframe_fixture_data):
    test_pd_dataframe = dataframe_fixture_data["test_pd_dataframe"]
    data_manager = DataManager(storage_type=storage_type, file_format=file_format)
    sid = data_manager.store(test_pd_dataframe)
    loaded_df = data_manager.load(sid)

    pd.testing.assert_frame_equal(loaded_df, test_pd_dataframe)


@pytest.mark.parametrize("storage_type", ['in_memory', 'filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_load(storage_type, file_format, dataframe_fixture_data):
    test_cudf_dataframe = dataframe_fixture_data["test_cudf_dataframe"]
    data_manager = DataManager(storage_type=storage_type, file_format=file_format)
    sid = data_manager.store(test_cudf_dataframe)
    loaded_df = data_manager.load(sid)

    pd.testing.assert_frame_equal(loaded_df, test_cudf_dataframe.to_pandas())


@pytest.mark.parametrize("storage_type", ['in_memory', 'filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_load_non_existent_source_id(storage_type, file_format):
    data_manager = DataManager(storage_type=storage_type, file_format=file_format)

    try:
        data_manager.load(uuid.uuid4())
        pytest.fail('Expected KeyError to be raised. (Source ID does not exist.')
    except KeyError:
        pass


@pytest.mark.parametrize("storage_type", ['in_memory', 'filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_get_num_rows(storage_type, file_format, dataframe_fixture_data):
    test_pd_dataframe = dataframe_fixture_data["test_pd_dataframe"]
    data_manager = DataManager(storage_type=storage_type, file_format=file_format)
    sid = data_manager.store(test_pd_dataframe)
    num_rows = data_manager.get_record(sid).num_rows
    assert (num_rows == len(test_pd_dataframe))


@pytest.mark.parametrize("storage_type", ['in_memory', 'filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_source_property(storage_type, file_format, dataframe_fixture_data):
    test_cudf_dataframe = dataframe_fixture_data["test_cudf_dataframe"]
    data_manager = DataManager(storage_type=storage_type, file_format=file_format)
    sid = data_manager.store(test_cudf_dataframe)  # noqa pylint: disable=unused-variable
    data_records = data_manager.records

    assert (len(data_records) == 1)

    for key, value in data_records.items():  # pylint: disable=unused-variable
        assert (value._storage_type == storage_type)
        if (storage_type == 'in_memory'):
            assert (isinstance(value.data, pd.DataFrame))
        elif (storage_type == 'filesystem'):
            assert (isinstance(value.data, pd.DataFrame))


@pytest.mark.parametrize("storage_type", ['in_memory', 'filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_store_from_existing_file_path(storage_type, file_format, dataframe_fixture_data):
    test_parquet_filepath = dataframe_fixture_data["test_parquet_filepath"]
    test_csv_filepath = dataframe_fixture_data["test_csv_filepath"]
    test_cudf_dataframe = dataframe_fixture_data["test_cudf_dataframe"]

    data_manager = DataManager(storage_type=storage_type, file_format=file_format)
    if (file_format == 'parquet'):
        sid = data_manager.store(test_parquet_filepath)
    elif (file_format == 'csv'):
        sid = data_manager.store(test_csv_filepath)
    else:
        sid = None

    loaded_df = data_manager.load(sid)
    assert (loaded_df.equals(test_cudf_dataframe.to_pandas()))


if __name__ == '__main__':
    unittest.main()
