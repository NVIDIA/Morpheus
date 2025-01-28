# Copyright (c) 2022-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Morpheus pipeline module for fetching files and emitting them as DataFrames."""

import hashlib
import json
import logging
import os
import time
import typing
from functools import partial

import fsspec
import pandas as pd

from morpheus.common import FileTypes
from morpheus.io.deserializers import read_file_to_df
from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.column_info import PreparedDFInfo
from morpheus.utils.column_info import process_dataframe
from morpheus.utils.downloader import Downloader
from morpheus.utils.downloader import DownloadMethods

logger = logging.getLogger(__name__)


def single_object_to_dataframe(file_object: fsspec.core.OpenFile,
                               schema: DataFrameInputSchema,
                               file_type: FileTypes,
                               filter_null: bool,
                               parser_kwargs: dict) -> pd.DataFrame:
    """
    Converts a file object into a Pandas DataFrame with optional preprocessing.

    Parameters
    ----------
    file_object : `fsspec.core.OpenFile`
        A file object, typically from a remote storage system.
    schema : `morpheus.utils.column_info.DataFrameInputSchema`
        A schema defining how to process the data.
    file_type : `morpheus.common.FileTypes`
        The type of the file being processed (e.g., CSV, Parquet).
    filter_null : bool
        Flag to indicate whether to filter out null values.
    parser_kwargs : dict
        Additional keyword arguments to pass to the file parser.

    Returns
    -------
        pd.DataFrame: The resulting Pandas DataFrame after processing and optional preprocessing.
    """

    retries = 0
    df = None
    while (retries < 2):
        try:
            with file_object as f:
                df = read_file_to_df(f,
                                     file_type,
                                     filter_nulls=filter_null,
                                     df_type="pandas",
                                     parser_kwargs=parser_kwargs)

            break
        except Exception as e:
            if (retries < 2):
                logger.warning("Error fetching %s: %s\nRetrying...", file_object, e)
                retries += 1

    # Optimistaclly prep the dataframe (Not necessary since this will happen again in process_dataframe, but it
    # increases performance significantly)
    if (schema.prep_dataframe is not None):
        prepared_df_info: PreparedDFInfo = schema.prep_dataframe(df)

    return prepared_df_info.df


class FileToDFController:
    """
    Controller class for converting file objects to Pandas DataFrames with optional preprocessing.

    Parameters
    ----------
    schema : DataFrameInputSchema
        A schema defining how to process the data.
    filter_null : bool
        Flag to indicate whether to filter out null values.
    file_type : FileTypes
        The type of the file being processed (e.g., CSV, Parquet).
    parser_kwargs : dict
        Additional keyword arguments to pass to the file parser.
    cache_dir : str
        Directory where cache will be stored.
    timestamp_column_name : str
        Name of the timestamp column.
    download_method : typing.Union[DownloadMethods, str], optional, default = DownloadMethods.DASK_THREAD
        The download method to use, if the `MORPHEUS_FILE_DOWNLOAD_TYPE` environment variable is set, it takes
        presedence.
    """

    def __init__(self,
                 schema: DataFrameInputSchema,
                 filter_null: bool,
                 file_type: FileTypes,
                 parser_kwargs: dict,
                 cache_dir: str,
                 timestamp_column_name: str,
                 download_method: typing.Union[DownloadMethods, str] = DownloadMethods.DASK_THREAD):

        self._schema = schema
        self._file_type = file_type
        self._filter_null = filter_null
        self._parser_kwargs = {} if parser_kwargs is None else parser_kwargs
        self._cache_dir = os.path.join(cache_dir, "file_cache")
        self._timestamp_column_name = timestamp_column_name

        self._downloader = Downloader(download_method=download_method)

    def _get_or_create_dataframe_from_batch(
            self, file_object_batch: typing.Tuple[fsspec.core.OpenFiles, int]) -> typing.Tuple[pd.DataFrame, bool]:

        if (not file_object_batch):
            raise RuntimeError("No file objects to process")

        file_list = file_object_batch[0]
        batch_count = file_object_batch[1]

        file_system: fsspec.AbstractFileSystem = file_list.fs

        # Create a list of dictionaries that only contains the information we are interested in hashing. `ukey` just
        # hashes all of the output of `info()` which is perfect
        hash_data = [{"ukey": file_system.ukey(file_object.path)} for file_object in file_list]

        # Convert to base 64 encoding to remove - values
        objects_hash_hex = hashlib.md5(json.dumps(hash_data, sort_keys=True).encode()).hexdigest()

        batch_cache_location = os.path.join(self._cache_dir, "batches", f"{objects_hash_hex}.pkl")

        # Return the cache if it exists
        if (os.path.exists(batch_cache_location)):
            output_df = pd.read_pickle(batch_cache_location)
            output_df["batch_count"] = batch_count
            output_df["origin_hash"] = objects_hash_hex

            return (output_df, True)

        # Cache miss
        download_method_func = partial(single_object_to_dataframe,
                                       file_type=self._file_type,
                                       schema=self._schema,
                                       filter_null=self._filter_null,
                                       parser_kwargs=self._parser_kwargs)

        download_buckets = file_list

        # Loop over dataframes and concat into one
        try:
            dfs = self._downloader.download(download_buckets, download_method_func)
        except Exception:
            logger.exception("Failed to download logs. Error: ", exc_info=True)
            raise

        if (dfs is None or len(dfs) == 0):
            raise ValueError("No logs were downloaded")

        output_df: pd.DataFrame = pd.concat(dfs)

        output_df = process_dataframe(df_in=output_df, input_schema=self._schema)

        # Finally sort by timestamp and then reset the index
        output_df.sort_values(by=[self._timestamp_column_name], inplace=True)

        output_df.reset_index(drop=True, inplace=True)

        # Save dataframe to cache future runs
        os.makedirs(os.path.dirname(batch_cache_location), exist_ok=True)

        try:
            output_df.to_pickle(batch_cache_location)
        except Exception:
            logger.warning("Failed to save batch cache. Skipping cache for this batch.", exc_info=True)

        output_df["batch_count"] = batch_count
        output_df["origin_hash"] = objects_hash_hex

        return (output_df, False)

    def convert_to_dataframe(self, file_object_batch: typing.Tuple[fsspec.core.OpenFiles, int]) -> pd.DataFrame:
        """
        Convert a batch of file objects to a DataFrame.

        Parameters
        ----------
        file_object_batch : typing.Tuple[fsspec.core.OpenFiles, int]
            A batch of file objects and batch count.

        Returns
        -------
        pd.DataFrame
            The resulting DataFrame.
        """

        if (not file_object_batch):
            return None

        start_time = time.time()

        try:
            output_df, cache_hit = self._get_or_create_dataframe_from_batch(file_object_batch)

            duration = (time.time() - start_time) * 1000.0

            if (output_df is not None and logger.isEnabledFor(logging.DEBUG)):
                logger.debug("S3 objects to DF complete. Rows: %s, Cache: %s, Duration: %s ms, Rate: %s rows/s",
                             len(output_df),
                             "hit" if cache_hit else "miss",
                             duration,
                             len(output_df) / (duration / 1000.0))

            return output_df
        except Exception:
            logger.exception("Error while converting S3 buckets to DF.")
            raise

    def close(self):
        """
        Close the resources used by the controller.
        """
        self._downloader.close()
