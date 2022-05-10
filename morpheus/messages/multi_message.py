import dataclasses
import typing

import cudf

import morpheus._lib.messages as neom
import morpheus.messages as _messages
from morpheus.messages.message_base import MessageData


@dataclasses.dataclass
class MultiMessage(MessageData, cpp_class=neom.MultiMessage):
    """
    This class holds data for multiple messages at a time. To avoid copying data for slicing operations, it
    holds a reference to a batched metadata object and stores the offset and count into that batch.

    Parameters
    ----------
    meta : `MessageMeta`
        Deserialized messages metadata for large batch.
    mess_offset : int
        Offset into the metadata batch.
    mess_count : int
        Messages count.

    """
    meta: _messages.MessageMeta = dataclasses.field(repr=False)
    mess_offset: int
    mess_count: int

    @property
    def id_col(self):
        """
        Returns ID column values from `morpheus.pipeline.messages.MessageMeta.df`.

        Returns
        -------
        pandas.Series
            ID column values from the dataframe.

        """
        return self.get_meta("ID")

    @property
    def id(self) -> typing.List[int]:
        """
        Returns ID column values from `morpheus.pipeline.messages.MessageMeta.df` as list.

        Returns
        -------
        List[int]
            ID column values from the dataframe as list.

        """

        return self.get_meta_list("ID")

    @property
    def timestamp(self) -> typing.List[int]:
        """
        Returns timestamp column values from morpheus.messages.MessageMeta.df as list.

        Returns
        -------
        List[int]
            Timestamp column values from the dataframe as list.

        """

        return self.get_meta_list("timestamp")

    def get_meta(self, columns: typing.Union[None, str, typing.List[str]] = None):
        """
        Return column values from `morpheus.pipeline.messages.MessageMeta.df`.

        Parameters
        ----------
        columns : typing.Union[None, str, typing.List[str]]
            Input column names. Returns all columns if `None` is specified. When a string is passed, a `Series` is
            returned. Otherwise a `Dataframe` is returned.

        Returns
        -------
        Series or Dataframe
            Column values from the dataframe.

        """

        idx = self.meta.df.index[self.mess_offset:self.mess_offset + self.mess_count]

        if (isinstance(idx, cudf.RangeIndex)):
            idx = slice(idx.start, idx.stop - 1, idx.step)

        if (columns is None):
            return self.meta.df.loc[idx, :]
        else:
            # If its a str or list, this is the same
            return self.meta.df.loc[idx, columns]

    def get_meta_list(self, col_name: str = None):
        """
        Return a column values from morpheus.messages.MessageMeta.df as a list.

        Parameters
        ----------
        col_name : str
            Column name in the dataframe.

        Returns
        -------
        List[str]
            Column values from the dataframe.

        """

        return self.get_meta(col_name).to_list()

    def set_meta(self, columns: typing.Union[None, str, typing.List[str]], value):
        """
        Set column values to `morpheus.pipelines.messages.MessageMeta.df`.

        Parameters
        ----------
        columns : typing.Union[None, str, typing.List[str]]
            Input column names. Sets the value for the corresponding column names. If `None` is specified, all columns
            will be used. If the column does not exist, a new one will be created.
        value : Any
            Value to apply to the specified columns. If a single value is passed, it will be broadcast to all rows. If a
            `Series` or `Dataframe` is passed, rows will be matched by index.

        """
        if (columns is None):
            # Set all columns
            self.meta.df.loc[self.meta.df.index[self.mess_offset:self.mess_offset + self.mess_count], :] = value
        else:
            # If its a single column or list of columns, this is the same
            self.meta.df.loc[self.meta.df.index[self.mess_offset:self.mess_offset + self.mess_count], columns] = value

    def get_slice(self, start, stop):
        """
        Returns sliced batches based on offsets supplied. Automatically calculates the correct `mess_offset`
        and `mess_count`.

        Parameters
        ----------
        start : int
            Start offset address.
        stop : int
            Stop offset address.

        Returns
        -------
        `MultiInferenceMessage`
            A new `MultiInferenceMessage` with sliced offset and count.

        """
        return MultiMessage(meta=self.meta, mess_offset=start, mess_count=stop - start)
