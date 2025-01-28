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
import uuid
from typing import Any
from typing import Optional
from typing import Union

import fsspec
import pandas as pd

import cudf

from morpheus.io.data_record import DataRecord


class DataManager():
    """
    DataManager class to manage the storage and retrieval of files
    using either in-memory or filesystem storage.
    """

    VALID_STORAGE_TYPES = ('in_memory', 'filesystem')
    VALID_FILE_FORMATS = ('parquet', 'csv')

    def __init__(self, storage_type: str = 'in_memory', file_format: str = 'parquet'):
        """
        Initialize the DataManager instance.

        :param storage_type: Specifies the storage type to be used. Can be either 'in_memory' or 'filesystem'.
        :param file_format: Specifies the file format to be used. Can be either 'parquet' or 'csv'.
        """

        # Define these early so that they are defined even if we raise an exception, this ensures that we don't get an
        # attribute error in the __del__ method.
        self._storage_dir = None
        self._storage_type = None

        if (storage_type not in self.VALID_STORAGE_TYPES):
            raise ValueError(f"Invalid storage_type '{storage_type}'")

        if (file_format not in self.VALID_FILE_FORMATS):
            raise ValueError(f"Invalid file_format '{file_format}'")

        self._dirty = True
        self._file_format = file_format
        self._fs = fsspec.filesystem('file')
        self._manifest = {}
        self._records = {}
        self._storage_type = storage_type
        self._total_rows = 0

        if (storage_type == 'filesystem'):
            self._storage_dir = tempfile.mkdtemp()

    def __contains__(self, item: Any) -> bool:
        return item in self._records

    def __del__(self):
        if (self._storage_type == 'filesystem'):
            shutil.rmtree(self._storage_dir)

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self):
        return (f"DataManager(records={self.num_rows}, "
                f"storage_type={self._storage_type!r}, "
                f"storage directory={self._storage_dir!r})")

    def __str__(self):
        return (f"DataManager with {self.num_rows} records, "
                f"storage type: {self._storage_type}, "
                f"storage directory: {self._storage_dir}")

    @property
    def manifest(self) -> dict:
        """
        Retrieve a mapping of UUIDs to their filenames or labels.

        :return: A dictionary containing UUID to filename/label mappings.
        """

        if (self._dirty):
            self._manifest = {source_id: data_record.backing_source for source_id, data_record in self._records.items()}

        return self._manifest

    @property
    def num_rows(self) -> int:
        """
        Get the number of rows in a source given its source ID.
        :return:
        """

        return self._total_rows

    @property
    def records(self):
        return self._records

    @property
    def storage_type(self) -> str:
        """
        Get the storage type used by the DataManager instance.

        :return: Storage type as a string.
        """

        return self._storage_type

    def get_record(self, source_id: uuid.UUID) -> DataRecord:
        """
        Get a DataRecord instance given a source ID.

        :param source_id: UUID of the source to be retrieved.
        :return: DataRecord instance.
        """

        if source_id not in self._records:
            raise KeyError(f"Source ID '{source_id}' not found.")

        return self._records[source_id]

    def load(self, source_id: uuid.UUID) -> cudf.DataFrame:
        """
        Load a cuDF DataFrame given a source ID.

        :param source_id: UUID of the source to be loaded.
        :return: Loaded cuDF DataFrame.
        """

        if source_id not in self._records:
            raise KeyError(f"Source ID '{source_id}' not found.")

        data_record = self._records[source_id]

        return data_record.load()

    def store(self,
              data_source: Union[cudf.DataFrame, pd.DataFrame, str],
              copy_from_source: bool = False,
              data_label: Optional[str] = None) -> uuid.UUID:
        """
        Store a DataFrame or file path as a source and return the source ID.

        :param data_source: DataFrame or file path to store as a source.
        :param copy_from_source: Whether to copy the data on disk when the input is a file path and the storage type is
            'filesystem'.
        :param data_label: Optional label for the stored data.
        :return: UUID of the stored source.
        """

        tracking_id = uuid.uuid4()
        while (tracking_id in self._records):
            # Ensure that the tracking ID is unique.
            tracking_id = uuid.uuid4()

        if (self._storage_type == 'filesystem'):
            data_label = os.path.join(self._storage_dir, f"{tracking_id}.{self._file_format}")
        else:
            data_label = data_label or f'dataframe_{tracking_id}'

        data_record = DataRecord(data_source=data_source,
                                 data_label=data_label,
                                 storage_type=self.storage_type,
                                 file_format=self._file_format,
                                 copy_from_source=copy_from_source)

        self._total_rows += len(data_record)
        self._records[tracking_id] = data_record
        self._dirty = True

        return tracking_id

    def remove(self, source_id: uuid.UUID) -> None:
        """
        Remove a source using its source ID.

        :param source_id: UUID of the source to be removed.
        """

        if (source_id not in self._records):
            raise KeyError(f"Source ID '{source_id}' does not exist.")

        row_count = len(self._records[source_id])
        del self._records[source_id]

        self._total_rows -= row_count
        self._dirty = True
