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

import typing
from typing import Union

import pandas as pd

import cudf

from morpheus.io.data_storage import FileSystemStorage
from morpheus.io.data_storage import InMemoryStorage


class DataRecord:
    """Class for managing data records in different storage types and formats.

    Attributes:
        VALID_STORAGE_TYPES (tuple): Allowed storage types.
        VALID_FILE_FORMATS (tuple): Allowed file formats.
    """

    VALID_STORAGE_TYPES = ('in_memory', 'filesystem')
    VALID_FILE_FORMATS = ('parquet', 'csv')

    def __init__(self,
                 data_source: Union[pd.DataFrame, cudf.DataFrame, str],
                 data_label: str,
                 storage_type: str,
                 file_format: str = "parquet",
                 copy_from_source: bool = False):
        """Initialize a DataRecord instance.

        Args:
            data_source (Union[io.BytesIO, str]): Data source, either a file path or Dataframe.
            data_label (str): Label for the data record.
            storage_type (str): Storage type, either 'in_memory' or 'filesystem'.
            file_format (str): File format, either 'parquet' or 'csv'.
            copy_from_source (bool, optional): If True, copy data from the source. Defaults to False.
        """

        self._copy_from_source = copy_from_source
        self._data_label = data_label  # This will be the full path to the file or the name of the BytesIO object
        self._file_format = file_format
        self._storage = None
        self._storage_type = storage_type

        if (self._storage_type == 'in_memory'):
            self._storage = InMemoryStorage(file_format=self._file_format)
        elif (self._storage_type == 'filesystem'):
            self._storage = FileSystemStorage(file_path=self._data_label, file_format=self._file_format)
        else:
            raise ValueError(f"Invalid storage_type'{storage_type}'")

        self._storage.store(data_source, self._copy_from_source)

    def __del__(self):
        """Delete the DataRecord instance."""

        if (self._storage is not None):
            self._storage.delete()

    def __len__(self) -> int:
        """Return the number of rows in the data record."""

        return self.num_rows

    def __repr__(self) -> str:
        """Return a string representation of the DataRecord instance."""

        return (f"DataRecord(data_label={self._data_label!r}, "
                f"storage_type={self._storage_type!r}, "
                f"file_format={self._file_format!r}, "
                f"num_rows={self.num_rows}, "
                f"owner={self._storage.owner})")

    def __str__(self) -> str:
        """Return a string representation of the DataRecord instance."""

        return (f"DataRecord with label '{self._data_label}', "
                f"stored as {self._storage_type}, "
                f"file format: {self._file_format}, "
                f"number of rows: {self.num_rows}")

    def load(self) -> cudf.DataFrame:
        """Load a cuDF DataFrame from the DataRecord.

        Returns:
            cudf.DataFrame: Loaded cuDF DataFrame.
        """

        return self._storage.load()

    @property
    def data_label(self) -> str:
        """Get the label for the data record.

        Returns:
            str: Label for the data record.
        """

        return self._data_label

    @property
    def backing_source(self) -> str:
        """Get the backing file for the data record.

        Returns:
            str: Backing file for the data record.
        """

        return self._storage.backing_source

    @property
    def data(self) -> typing.Union[cudf.DataFrame, pd.DataFrame]:
        """Get the data associated with the data record.

        Returns:
            Any: Data associated with the data record.
        """

        return self._storage.load()

    @property
    def format(self) -> str:
        """
        Get the file format of the data record.
        Returns:
        str: File format of the data record.
        """
        return self._file_format

    @property
    def num_rows(self) -> int:
        """Get the number of rows in the data record.

        Returns:
        int: Number of rows in the data record.
        """

        return self._storage.num_rows
