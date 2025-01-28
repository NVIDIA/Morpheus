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

import io
from typing import Union

import pandas as pd

import cudf

from morpheus.io.data_storage_interface import RecordStorageInterface


class InMemoryStorage(RecordStorageInterface):
    """A class to manage storage of data in memory."""

    def __init__(self, file_format: str):
        """Initialize InMemoryStorage with a label and file format.

        Args:
            data_label: The label for the data.
            file_format: The format of the file ('csv' or 'parquet').
        """

        super().__init__(file_format)

        self._backing_source = "IO Buffer"
        self._data = io.BytesIO()
        self._num_rows = 0
        self._owner = True

        if self._file_format == 'csv':
            self._data_reader = pd.read_csv
            self._data_writer = lambda df, buffer: df.to_csv(buffer, index=False, header=True)
        elif self._file_format == 'parquet':
            self._data_reader = pd.read_parquet
            self._data_writer = lambda df, buffer: df.to_parquet(buffer, index=False)
        else:
            raise NotImplementedError(f"File format {self._file_format} is not supported.")

    def load(self) -> pd.DataFrame:
        """Load data from the buffer.

        Returns:
            The loaded data as a pandas DataFrame.
        """

        self._data.seek(0)
        return self._data_reader(self._data)

    def delete(self) -> None:
        """Delete the data in the buffer if this instance is the owner."""

        if self._owner:
            self._data.close()

    # yapf: disable -- yapf wants to put 'store' on a single line, flake8 says this is an error
    def store(self, data_source: Union[pd.DataFrame, cudf.DataFrame, str],
              copy_from_source: bool = True) -> None:  # pylint: disable=unused-argument
        # yapf: enable
        """Store data in the buffer.

        Args:
            data_source: The data to store. Can be a pandas.DataFrame, cudf.DataFrame, or a file path.
            copy_from_source: If True, data_source is assumed to be a file path and the data will be copied
            from this file to the buffer. If False, the buffer will simply be updated to point to data_source
            (which is assumed to be a file path).
        """

        self._data = io.BytesIO()

        if isinstance(data_source, str):
            data_source = self._data_reader(data_source)

        self._data_writer(data_source, self._data)
        self._num_rows = len(data_source)

    @property
    def backing_source(self) -> str:
        """Get the backing source.

        Returns:
            The name of the backing source.
        """

        return self._backing_source

    @property
    def num_rows(self) -> int:
        """Get the number of rows in the data.

        Returns:
            The number of rows in the data.
        """

        return self._num_rows

    @property
    def owner(self) -> bool:
        """Get whether this instance is the owner of the data in the buffer.

        Returns:
            True if this instance is the owner of the data in the buffer, False otherwise.
        """

        return self._owner
