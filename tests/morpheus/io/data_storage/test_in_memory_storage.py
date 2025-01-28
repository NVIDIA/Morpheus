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

import pandas as pd
import pytest

from morpheus.io.data_storage import InMemoryStorage


def test_in_memory_storage_invalid_file_format():
    # Test that an error is raised when an unsupported file format is used
    with pytest.raises(NotImplementedError):
        InMemoryStorage('txt')


def test_in_memory_storage_store_df():
    # Test that a DataFrame can be stored and loaded correctly
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    storage = InMemoryStorage('csv')
    storage.store(df)
    assert storage.num_rows == 3
    assert isinstance(storage.load(), pd.DataFrame)
    assert storage.load().equals(df)


def test_in_memory_storage_store_file():
    # Test that a file can be stored and loaded correctly
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    storage = InMemoryStorage('csv')
    storage.store(df)
    assert storage.num_rows == 3
    assert isinstance(storage.load(), pd.DataFrame)
    assert storage.load().equals(df)


def test_in_memory_storage_delete():
    # Test that delete closes the buffer
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    storage = InMemoryStorage('csv')
    storage.store(df)
    storage.delete()
    with pytest.raises(ValueError):
        storage.load()
