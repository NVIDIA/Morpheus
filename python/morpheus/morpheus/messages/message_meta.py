# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

import dataclasses
import logging
import threading
import typing
import warnings

import numpy as np
import pandas as pd

import morpheus._lib.messages as _messages
from morpheus.messages.message_base import MessageBase
from morpheus.utils import logger as morpheus_logger
from morpheus.utils.type_aliases import DataFrameType
from morpheus.utils.type_aliases import SeriesType

logger = logging.getLogger(__name__)


class MutableTableCtxMgr:
    """
    Context manager for editing the DataFrame held by a MessageMeta, ensures an editing lock aqcuired and released.
    Not intended to be used directly but is instead invoked via MessageMeta's `mutable_dataframe`.

    Examples
    --------
    >>> with meta.mutable_dataframe() as df:
    >>>     df['col'] = 5
    """

    ussage_error = ("Error attempting to use mutable_dataframe outside of context manager. Intended usage :\n"
                    "with message_meta.mutable_dataframe() as df:\n"
                    "    df['col'] = 5")

    def __init__(self, meta) -> None:
        self.__dict__['__meta'] = meta

    def __enter__(self) -> DataFrameType:
        meta = self.__dict__['__meta']
        meta._mutex.acquire()
        return meta._df

    def __exit__(self, exc_type, exc_value, traceback):
        self.__dict__['__meta']._mutex.release()

    def __getattr__(self, name):
        raise AttributeError(self.ussage_error)

    def __getitem__(self, key):
        raise AttributeError(self.ussage_error)

    def __setattr__(self, name, value):
        raise AttributeError(self.ussage_error)

    def __setitem__(self, key, value):
        raise AttributeError(self.ussage_error)


@dataclasses.dataclass(init=False)
class MessageMeta(MessageBase, cpp_class=_messages.MessageMeta):
    """
    This is a container class to hold batch deserialized messages metadata.

    Parameters
    ----------
    df : pandas.DataFrame
        Input rows in dataframe.

    """
    _df: DataFrameType = dataclasses.field(init=False)
    _mutex: threading.RLock = dataclasses.field(init=False, repr=False)

    def __init__(self, df: DataFrameType) -> None:
        super().__init__()
        if isinstance(df, MessageMeta):
            df = df.copy_dataframe()

        self._mutex = threading.RLock()
        self._df = df

    def _get_col_indexers(self, df, columns: typing.Union[None, str, typing.List[str]] = None):

        if (columns is None):
            columns = df.columns.to_list()
        elif (isinstance(columns, str)):
            # Convert a single string into a list so all versions return tables, not series
            columns = [columns]

        column_indexer = df.columns.get_indexer_for(columns)

        return column_indexer

    @property
    def df(self) -> DataFrameType:
        msg = ("Warning the df property returns a copy, please use the copy_dataframe method or the mutable_dataframe "
               "context manager to modify the DataFrame in-place instead.")

        warnings.warn(msg, DeprecationWarning)
        return self.copy_dataframe()

    def copy_dataframe(self) -> DataFrameType:
        return self._df.copy(deep=True)

    def mutable_dataframe(self):
        return MutableTableCtxMgr(self)

    @property
    def count(self) -> int:
        """
        Returns the number of messages in the batch.

        Returns
        -------
        int
            number of messages in the MessageMeta.df.
        """

        return len(self._df)

    def has_sliceable_index(self) -> bool:
        """
        Returns True if the underlying DataFrame's index is unique and monotonic. Sliceable indices have better
        performance since a range of rows can be specified by a start and stop index instead of requiring boolean masks.

        Returns
        -------
        bool
        """

        # Must be either increasing or decreasing with unique values to slice
        return self._df.index.is_unique and (self._df.index.is_monotonic_increasing
                                             or self._df.index.is_monotonic_decreasing)

    def ensure_sliceable_index(self) -> str:
        """
        Replaces the index in the underlying dataframe if the existing one is not unique and monotonic. The old index
        will be preserved in a column named `_index_{old_index.name}`. If `has_sliceable_index() == true`, this is a
        no-op.

        Returns
        -------
        str
            The name of the column with the old index or `None` if no changes were made
        """

        if (not self.has_sliceable_index()):

            # Reset the index preserving the original index in a new column
            with self.mutable_dataframe() as df:
                # We could have had a race condition between calling has_sliceable_index() and acquiring the mutex.
                # Perform a second check here while we hold the lock.
                if (not df.index.is_unique
                        or not (df.index.is_monotonic_increasing or df.index.is_monotonic_decreasing)):
                    logger.info("Non unique index found in dataframe, generating new index.")
                    df.index.name = "_index_" + (df.index.name or "")

                    old_index_name = df.index.name

                    df.reset_index(inplace=True)

                    return old_index_name

        return None

    def get_column_names(self) -> list[str]:
        return self._df.columns.to_list()

    def get_meta_range(self,
                       mess_offset: int,
                       message_count: int,
                       columns: typing.Union[None, str, typing.List[str]] = None):
        """
        Return column values from `morpheus.pipeline.messages.MessageMeta.df` from the specified start offset
        until the message count.

        Parameters
        ----------
        mess_offset : int
            Offset into the metadata batch.
        message_count : int
            Messages count.
        columns : typing.Union[None, str, typing.List[str]]
            Input column names. Returns all columns if `None` is specified. When a string is passed, a `Series` is
            returned. Otherwise a `Dataframe` is returned.

        Returns
        -------
        Series or Dataframe
            Column values from the dataframe.

        """

        idx = self._df.index[mess_offset:mess_offset + message_count]

        if (isinstance(idx, pd.RangeIndex)):
            idx = slice(idx.start, idx.stop - 1, idx.step)

        if (columns is None):
            return self._df.loc[idx, :]

        # If its a str or list, this is the same
        return self._df.loc[idx, columns]

    @typing.overload
    def get_data(self) -> DataFrameType:
        ...

    @typing.overload
    def get_data(self, columns: str) -> SeriesType:
        ...

    @typing.overload
    def get_data(self, columns: typing.List[str]) -> DataFrameType:
        ...

    def get_data(self, columns: typing.Union[None, str, typing.List[str]] = None):
        """
        Return column values from the underlying DataFrame.

        Parameters
        ----------
        columns : typing.Union[None, str, typing.List[str]]
            Input column names. Returns all columns if `None` is specified. When a string is passed, a `Series` is
            returned. Otherwise, a `Dataframe` is returned.

        Returns
        -------
        Series or Dataframe
            Column values from the dataframe.

        """

        with self.mutable_dataframe() as df:
            column_indexer = self._get_col_indexers(df, columns=columns)

            if (-1 in column_indexer):
                missing_columns = [columns[i] for i, index_value in enumerate(column_indexer) if index_value == -1]
                raise KeyError(f"Requested columns {missing_columns} does not exist in the dataframe")

            if (isinstance(columns, str) and len(column_indexer) == 1):
                # Make sure to return a series for a single column
                column_indexer = column_indexer[0]

            return df.iloc[:, column_indexer]

    def set_data(self, columns: typing.Union[None, str, typing.List[str]], value):
        """
        Set column values to the underlying DataFrame.

        Parameters
        ----------
        columns : typing.Union[None, str, typing.List[str]]
            Input column names. Sets the value for the corresponding column names. If `None` is specified, all columns
            will be used. If the column does not exist, a new one will be created.
        value : Any
            Value to apply to the specified columns. If a single value is passed, it will be broadcast to all rows. If a
            `Series` or `Dataframe` is passed, rows will be matched by index.

        """

        # Get exclusive access to the dataframe
        with self.mutable_dataframe() as df:
            # First try to set the values on just our slice if the columns exist
            column_indexer = self._get_col_indexers(df, columns=columns)

            # Check to see if we are adding a column. If so, we need to use df.loc instead of df.iloc
            if (-1 not in column_indexer):

                # If we only have one column, convert it to a series (broadcasts work with more types on a series)
                if (len(column_indexer) == 1):
                    column_indexer = column_indexer[0]

                try:
                    # Now update the slice
                    df.iloc[:, column_indexer] = value
                except (ValueError, TypeError):
                    # Try this as a fallback. Works better for strings. See issue #286
                    df[columns].iloc[:] = value

            else:
                # Columns should never be empty if we get here
                assert columns is not None

                # Now set the slice
                df.loc[:, columns] = value

    def get_slice(self, start, stop):
        """
        Returns a new MessageMeta with only the rows specified by start/stop.

        Parameters
        ----------
        start : int
            Start offset address.
        stop : int
            Stop offset address.

        Returns
        -------
        `MessageMeta`
            A new `MessageMeta` with sliced offset and count.
        """

        with self.mutable_dataframe() as df:
            return MessageMeta(df.iloc[start:stop])

    def _ranges_to_mask(self, df, ranges):
        mask = np.zeros(len(df), bool)

        for range_ in ranges:
            mask[range_[0]:range_[1]] = True

        return mask

    def copy_ranges(self, ranges: typing.List[typing.Tuple[int, int]]):
        """
        Perform a copy of the current message instance for the given `ranges` of rows.

        Parameters
        ----------
        ranges : typing.List[typing.Tuple[int, int]]
            Rows to include in the copy in the form of `[(`start_row`, `stop_row`),...]`
            The `stop_row` isn't included. For example to copy rows 1-2 & 5-7 `ranges=[(1, 3), (5, 8)]`

        Returns
        -------
        `MessageMeta`
            A new `MessageMeta` with only the rows specified by `ranges`.
        """

        with self.mutable_dataframe() as df:
            mask = self._ranges_to_mask(df, ranges=ranges)
            return MessageMeta(df.loc[mask, :])


@dataclasses.dataclass(init=False)
class UserMessageMeta(MessageMeta, cpp_class=None):
    """
    This class extends MessageMeta to also hold userid corresponding to batched metadata.

    Parameters
    ----------
    df : pandas.DataFrame
        Input rows in dataframe.
    user_id : str
        User id.

    """
    user_id: str = dataclasses.field(init=False)

    def __init__(self, df: pd.DataFrame, user_id: str) -> None:
        from morpheus.messages.control_message import ControlMessage
        morpheus_logger.deprecated_message_warning(UserMessageMeta, ControlMessage)
        super().__init__(df)
        self.user_id = user_id


@dataclasses.dataclass(init=False)
class AppShieldMessageMeta(MessageMeta, cpp_class=None):
    """
    This class extends MessageMeta to also hold source corresponding to batched metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Input rows in dataframe.
    source : str
        Determines which source generated the snapshot messages.
    """
    source: str = dataclasses.field(init=False)

    def __init__(self, df: pd.DataFrame, source: str) -> None:
        from morpheus.messages.control_message import ControlMessage
        morpheus_logger.deprecated_message_warning(AppShieldMessageMeta, ControlMessage)
        super().__init__(df)
        self.source = source
