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
"""DatasetManager class for loading and caching test datasets as DataFrames."""

import logging
import os
import random
import typing
import warnings

import cupy as cp
import pandas as pd

import cudf as cdf  # rename to avoid clash with property method

from _utils import TEST_DIRS
from _utils import assert_results
from morpheus.io.deserializers import read_file_to_df
from morpheus.utils import compare_df
from morpheus.utils.type_aliases import DataFrameModule
from morpheus.utils.type_aliases import DataFrameType
from morpheus.utils.type_aliases import SeriesType


class DatasetManager:
    """
    Helper class for loading and caching test datasets as DataFrames, along with some common manipulation methods.

    Parameters
    ----------
    df_type : DataFrameTypeStr
        Type of DataFrame to return unless otherwise explicitly specified.
    """

    __df_cache: dict[tuple[DataFrameModule, str], DataFrameType] = {}

    # Values in `__instances` are instances of `DatasetLoader`
    __instances: dict[DataFrameModule, "DatasetManager"] = {}

    # Explicitly using __new__ instead of of an __init__ to implement this as a singleton for each dataframe type.
    # Initialization is also being performed here instead of an __init__ method as an __init__ method would be re-run
    # the __init__ on the singleton instance for each cache hit.
    def __new__(cls, df_type: DataFrameModule):
        """Returns the singleton instance of `DatasetManager` for the specified `df_type`."""
        try:
            return cls.__instances[df_type]
        except KeyError:
            instance = super().__new__(cls)
            instance._default_df_type = df_type
            cls.__instances[df_type] = instance
            return instance

    @staticmethod
    def get_alt_df_type(df_type: DataFrameModule) -> DataFrameModule:
        """Returns the other possible df type."""
        return 'cudf' if df_type == 'pandas' else 'pandas'

    def clear(self):
        """Clears the cache"""
        self.__df_cache.clear()

    def get_df(self,
               file_path: str,
               df_type: DataFrameModule = None,
               no_cache: bool = False,
               **reader_kwargs) -> DataFrameType:
        """
        Fetch a DataFrame specified from `file_path`. If `file_path` is not an absolute path, it is assumed to be
        relative to the `test/tests_data` dir. If a DataFrame matching both `file_path` and `df_type` has already been
        fetched, then a cached copy will be returned. In the event that a DataFrame matching `file_path` but not
        `df_type` exists in the cache, then the cached copy will be cast to the appropriate type, stored in the cache
        and then returned

        Passing values to `reader_kwargs` will cause the cache to by bypassed
        """
        if len(reader_kwargs) and not no_cache:
            logger = logging.getLogger(f"morpheus.{__name__}")
            logger.warning("Setting specific `reader_kwargs` requires bypassing the cache. "
                           "Set `no_cache=True` to avoid this warning.")
            no_cache = True

        abs_path = os.path.abspath(file_path)
        if abs_path != os.path.normpath(file_path):
            # Relative paths are assumed to be relative to the `tests/tests_data` dir
            full_path = os.path.abspath(os.path.join(TEST_DIRS.tests_data_dir, file_path))
        else:
            full_path = abs_path

        if df_type is None:
            df_type = self.default_df_type

        df = None
        if not no_cache:
            df = self.__df_cache.get((df_type, full_path))

        if df is None:
            alt_df = None
            if not no_cache:
                # If it isn't in the cache, but we have a cached copy in another DF format use it instead of re-reading
                alt_df_type = self.get_alt_df_type(df_type=df_type)
                alt_df = self.__df_cache.get((alt_df_type, full_path))

            if alt_df is not None:
                if alt_df_type == 'cudf':
                    df = alt_df.to_pandas()
                else:
                    df = cdf.DataFrame.from_pandas(alt_df)
            else:
                df = read_file_to_df(full_path, df_type=df_type, **reader_kwargs)

            if not no_cache:
                self.__df_cache[(df_type, full_path)] = df

        return df.copy(deep=True)

    def __getitem__(self, item: str | tuple[str] | tuple[str, DataFrameModule]) -> DataFrameType:
        """Implements `__getitem__` to allow for fetching DataFrames using the `[]` operator."""
        if not isinstance(item, tuple):
            item = (item, )

        return self.get_df(*item)

    @property
    def cudf(self):
        """Returns the singleton instance of `DatasetManager` for cudf DataFrames."""
        return DatasetManager(df_type='cudf')

    @property
    def pandas(self):
        """Returns the singleton instance of `DatasetManager` for pandas DataFrames."""
        return DatasetManager(df_type='pandas')

    @property
    def default_df_type(self):
        """Returns the default DataFrame type for this instance of `DatasetManager`."""
        return self._default_df_type

    @property
    def df_class(self):
        if self.default_df_type == 'cudf':
            cls = cdf.DataFrame
        else:
            cls = pd.DataFrame

        return cls

    @staticmethod
    def repeat(df: DataFrameType, repeat_count: int = 2, reset_index: bool = True) -> DataFrameType:
        """Returns a DF consisting of `repeat_count` copies of the original."""
        if isinstance(df, pd.DataFrame):
            concat_fn = pd.concat
        else:
            concat_fn = cdf.concat

        repeated_df = concat_fn([df for _ in range(repeat_count)])

        if reset_index:
            repeated_df = repeated_df.reset_index(inplace=False, drop=True)

        return repeated_df

    @staticmethod
    def replace_index(df: DataFrameType, replace_ids: dict[int, int]) -> DataFrameType:
        """Return a new DataFrame's where we replace some index values with others."""
        return df.rename(index=replace_ids)

    @classmethod
    def dup_index(cls, df: DataFrameType, count: int = 1) -> DataFrameType:
        """Randomly duplicate `count` entries in a DataFrame's index"""
        assert count * 2 <= len(df), "Count must be less than half the number of rows."

        # Sample 2x the count. One for the old ID and one for the new ID. Dont want duplicates so we use random.sample
        # (otherwise you could get less duplicates than requested if two IDs just swap)
        dup_ids = random.sample(df.index.values.tolist(), 2 * count)

        # Create a dictionary of old ID to new ID
        replace_dict = dict(zip(dup_ids[:count], dup_ids[count:]))

        # Return a new dataframe where we replace some index values with others
        return cls.replace_index(df, replace_dict)

    @staticmethod
    def _value_as_pandas(val: DataFrameType | SeriesType, assert_is_pandas=True):
        if (isinstance(val, (cdf.DataFrame, cdf.Series))):
            return val.to_pandas()

        if assert_is_pandas:
            assert isinstance(val, (pd.DataFrame, pd.Series)), type(val)

        return val

    @classmethod
    def _value_as_pandas_df(cls, val: DataFrameType | SeriesType, assert_is_pandas=True):
        pval = cls._value_as_pandas(val, assert_is_pandas=assert_is_pandas)
        if isinstance(pval, pd.Series):
            pval = pval.to_frame()

        return pval

    @classmethod
    def df_equal(cls, df_to_check: DataFrameType, val_to_check: typing.Any):
        """
        Compare a DataFrame against a validation dataset which can either be a DataFrame, Series or CuPy array. Returns
        True if they are equal.
        """
        # Comparisons work better in cudf so convert everything to that
        df_to_check = cls._value_as_pandas(df_to_check)

        if (isinstance(val_to_check, cp.ndarray)):
            val_to_check = val_to_check.get()
        else:
            val_to_check = cls._value_as_pandas(val_to_check, assert_is_pandas=False)

        if (isinstance(val_to_check, (pd.DataFrame, pd.Series))):
            return df_to_check.equals(val_to_check)

        bool_df = df_to_check == val_to_check

        return bool(bool_df.all(axis=None))

    @classmethod
    def assert_df_equal(cls,
                        df_to_check: DataFrameType,
                        val_to_check: typing.Any,
                        assert_msg="Dataframes are not equal."):
        """
        Compare a DataFrame against a validation dataset which can either be a DataFrame, Series or CuPy array.
        Assert the result is true with an optional message.
        """
        assert cls.df_equal(df_to_check=df_to_check, val_to_check=val_to_check), assert_msg

    @classmethod
    def compare_df(cls, dfa: DataFrameType, dfb: DataFrameType, **compare_args):
        """Wrapper for `morpheus.utils.compare_df.compare_df`."""
        with warnings.catch_warnings():
            # Ignore performance warnings from pandas triggered by the comparison
            warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
            return compare_df.compare_df(cls._value_as_pandas_df(dfa), cls._value_as_pandas_df(dfb), **compare_args)

    @classmethod
    def assert_compare_df(cls, dfa: DataFrameType, dfb: DataFrameType, **compare_args):
        """Convenience method for calling `compare_df` and asserting that the results are equivalent."""
        assert_results(cls.compare_df(dfa, dfb, **compare_args))
