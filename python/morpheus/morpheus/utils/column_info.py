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

import dataclasses
import json
import logging
import re
import typing
from datetime import datetime
from functools import partial

import pandas as pd

from morpheus.utils.type_aliases import DataFrameType
from morpheus.utils.type_utils import is_cudf_type

logger = logging.getLogger(f"morpheus.{__name__}")

DEFAULT_DATE = '1970-01-01T00:00:00.000000+00:00'


# Note(Devin): Proxying this for backwards compatibility. Had to move the primary definition to avoid circular imports.
def process_dataframe(df_in: DataFrameType, input_schema) -> pd.DataFrame:
    """
    Processes a dataframe according to the given schema.

    Parameters
    ----------
    df_in : pandas.DataFrame or cudf.DataFrame
        The input dataframe to process.
    input_schema : object
        The schema used to process the dataframe.

    Returns
    -------
    pandas.DataFrame
        The processed dataframe.

    """

    from morpheus.utils import schema_transforms  # pylint: disable=cyclic-import
    return schema_transforms.process_dataframe(df_in, input_schema)


def create_increment_col(df: pd.DataFrame,
                         column_name: str,
                         groupby_column: str = "username",
                         timestamp_column: str = "timestamp",
                         period: str = "D") -> pd.Series:
    """
    Create a new integer column counting unique occurrences of values in `column_name` grouped per-day using the
    timestamp values in `timestamp_column` and then grouping by `groupby_column` returning incrementing values starting
    at `1`.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe.
    column_name : str
        Name of the column in which unique occurrences are counted.
    groupby_column : str, default "username"
        The column to group by.
    timestamp_column : str, default "timestamp"
        The column containing timestamp values.
    period: str, default "D"
        The period to group by.

    Returns
    -------
    pandas.Series
        The new column with incrementing values.

    """

    # Ensure we are pandas for this
    if (not isinstance(df, pd.DataFrame)):
        df = df.to_pandas()

    time_col = df[timestamp_column].fillna(pd.to_datetime(DEFAULT_DATE))

    per_day = time_col.dt.to_period(period)

    cat_col: pd.Series = df.groupby([per_day, groupby_column
                                     ])[column_name].transform(lambda x: pd.factorize(x.fillna("nan"))[0] + 1)

    increment_col = pd.concat([cat_col, df[groupby_column]],
                              axis=1).groupby([per_day, groupby_column
                                               ])[column_name].expanding(1).max().droplevel(0).droplevel(0)

    return increment_col.astype("int")


def column_listjoin(df: pd.DataFrame, col_name: str) -> pd.Series:
    """
    Returns the array series `df[col_name]` as flattened string series.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe from which to get the column.
    col_name : str
        The column to transform.

    Returns
    -------
    pandas.Series
        A series with the arrays in the column flattened to strings.

    """

    if col_name in df:
        return df[col_name].transform(lambda x: ",".join(x)).astype('string')

    return pd.Series(None, dtype='string')


@dataclasses.dataclass
class ColumnInfo:
    """Defines a single column and type-cast."""
    name: str
    dtype: str

    def __setattr__(self, name: str, value: typing.Any) -> None:

        # Convert the dtype to a string when its set to keep the type consistent.
        if (name == "dtype"):
            value = ColumnInfo.convert_pandas_dtype(value)

        super().__setattr__(name, value)

    @staticmethod
    def convert_pandas_dtype(dtype: str | type) -> str:
        """Return the pandas type string of column."""
        if ((isinstance(dtype, str) and dtype.startswith("datetime"))
                or (isinstance(dtype, type) and issubclass(dtype, datetime))):
            return "datetime64[ns]"

        if (isinstance(dtype, str)):
            return dtype

        return dtype.__name__

    def get_input_column_types(self) -> dict[str, str]:
        """Return a dictionary of input column names and types needed for processing. This is used for schema
        validation and should be overridden by subclasses."""
        return {self.name: self.dtype}

    def get_pandas_dtype(self) -> str:
        """Return the pandas type string for the currently set `dtype`."""

        # The type is already converted. This is maintained for backwards compatibility.
        return self.dtype

    def _process_column(self, df: pd.DataFrame) -> pd.Series:
        """
        Performs the processing of the ColumnInfo. Most subclasses should override this method.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to process.

        Returns
        -------
        pd.Series
            New series defined by this class
        """
        if (self.name not in df.columns):
            return pd.Series(None, index=df.index, dtype=self.get_pandas_dtype())

        return df[self.name].astype(self.get_pandas_dtype())


@dataclasses.dataclass
class CustomColumn(ColumnInfo):
    """
    Subclass of `ColumnInfo`, defines a column to be computed by a user-defined function `process_column_fn`.

    Attributes
    ----------
    process_column_fn : Callable
        A function that takes a DataFrame and returns a Series.

    Methods
    -------
    _process_column(df: pandas.DataFrame) -> pandas.Series
        Apply the `process_column_fn` to the DataFrame and return the result as a Series.

    """

    process_column_fn: typing.Callable

    # The columns that the custom function will use
    input_column_types: dict[str, str] = dataclasses.field(default_factory=dict)

    def get_input_column_types(self) -> dict[str, str]:
        """
        Return a dictionary of input column names and types needed for processing. This is used for schema
        validation and should be overridden by subclasses.
        """
        return self.input_column_types

    def _process_column(self, df: pd.DataFrame) -> pd.Series:
        """
        Apply the `process_column_fn` to the DataFrame and return the result as a Series.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to process.

        Returns
        -------
        pandas.Series
            The processed column.
        """
        return self.process_column_fn(df)


@dataclasses.dataclass
class RenameColumn(ColumnInfo):
    """
    Subclass of `ColumnInfo`, adds the ability to also perform a rename.

    Attributes
    ----------
    input_name : str
        The name of the input column.

    Methods
    -------
    _process_column(df: pandas.DataFrame) -> pandas.Series
        Rename the column and return it as a Series.

    """
    input_name: str

    def get_input_column_types(self) -> dict[str, str]:
        """Return a dictionary of input column names and types needed for processing. This is used for schema
        validation and should be overridden by subclasses."""
        return {self.input_name: self.dtype}

    def _process_column(self, df: pd.DataFrame) -> pd.Series:
        """
        Rename the column and return it as a Series.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the column.

        Returns
        -------
        pandas.Series
            The renamed column.
        """

        if (self.input_name not in df.columns):
            return pd.Series(None, index=df.index, dtype=self.get_pandas_dtype())

        return df[self.input_name].astype(self.get_pandas_dtype())


@dataclasses.dataclass
class BoolColumn(RenameColumn):
    """
    Subclass of `RenameColumn`, adds the ability to map a set custom values as boolean values.

    Attributes
    ----------
    value_map : Dict[str, bool]
        A mapping from input values to boolean values.
    true_value : str
        The value to map to True.
    false_value : str
        The value to map to False.
    true_values : List[str]
        A list of values to map to True.
    false_values : List[str]
        A list of values to map to False.

    Methods
    -------
    _process_column(df: pandas.DataFrame) -> pandas.Series
        Apply the mapping and return the result as a boolean Series.

    """

    value_map: typing.Dict[str, bool] = dataclasses.field(init=False, default_factory=dict)

    true_value: dataclasses.InitVar[str] = None
    false_value: dataclasses.InitVar[str] = None

    true_values: dataclasses.InitVar[typing.List[str]] = None
    false_values: dataclasses.InitVar[typing.List[str]] = None

    def get_input_column_types(self) -> dict[str, str]:
        """Return a dictionary of input column names and types needed for processing. This is used for schema
        validation and should be overridden by subclasses."""
        return {self.input_name: 'str'}

    def __post_init__(self,
                      true_value: str,
                      false_value: str,
                      true_values: typing.List[str],
                      false_values: typing.List[str]):
        assert self.dtype == ColumnInfo.convert_pandas_dtype(bool), "BoolColumn must have dtype 'bool'"

        if (true_value is not None):
            self.value_map.update({true_value: True})

        if (false_value is not None):
            self.value_map.update({false_value: False})

        if (true_values is not None):
            self.value_map.update({v: True for v in true_values})

        if (false_values is not None):
            self.value_map.update({v: False for v in false_values})

    def _process_column(self, df: pd.DataFrame) -> pd.Series:
        """
        Apply the mapping and return the result as a boolean Series.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the column.

        Returns
        -------
        pandas.Series
            The processed column as a boolean Series.
        """

        return df[self.input_name].map(self.value_map).astype(bool)


@dataclasses.dataclass
class DateTimeColumn(RenameColumn):
    """
    Subclass of `RenameColumn`, specific to casting UTC localized datetime values. When incoming values contain a
    time-zone offset string the values are converted to UTC, while values without a time-zone are assumed to be UTC.

    Methods
    -------
    _process_column(df: pandas.DataFrame) -> pandas.Series
        Convert the values in the column to datetime and return the result as a Series.

    """

    def get_input_column_types(self) -> dict[str, str]:
        """
        Return a dictionary of input column names and types needed for processing. This is used for schema
        validation and should be overridden by subclasses.
        """
        return {self.input_name: ColumnInfo.convert_pandas_dtype(str)}

    def _process_column(self, df: pd.DataFrame) -> pd.Series:
        """
        Convert the values in the column to datetime and return the result as a Series.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the column.

        Returns
        -------
        pandas.Series
            The processed column as a datetime Series.
        """

        dt_series = pd.to_datetime(df[self.input_name], utc=True)

        dtype = self.get_pandas_dtype()
        if dtype == 'datetime64[ns]':
            # avoid deprecation warning about using .astype to convert from a tz-aware type to a tz-naive type
            return dt_series.dt.tz_localize(None)

        return dt_series.astype(dtype)


@dataclasses.dataclass
class StringJoinColumn(RenameColumn):
    """
    Subclass of `RenameColumn`, converts incoming `list` values to string by joining by `sep`.

    Attributes
    ----------
    sep : str
        The separator to use when joining the values.

    Methods
    -------
    _process_column(df: pandas.DataFrame) -> pandas.Series
        Join the values in the column and return the result as a Series.

    """

    sep: str

    def _process_column(self, df: pd.DataFrame) -> pd.Series:
        """
        Join the values in the column and return the result as a Series.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the column.

        Returns
        -------
        pandas.Series
            The processed column as a string Series.
        """

        return super()._process_column(df).str.join(sep=self.sep)


@dataclasses.dataclass
class StringCatColumn(ColumnInfo):
    """
    Subclass of `ColumnInfo`, concatenates values from multiple columns into a new string column separated by `sep`.

    Attributes
    ----------
    input_columns : List[str]
        The columns to concatenate.
    sep : str
        The separator to use when joining the values.

    Methods
    -------
    _process_column(df: pandas.DataFrame) -> pandas.Series
        Concatenate the values from the input columns and return the result as a Series.

    """
    input_columns: typing.List[str]
    sep: str

    def get_input_column_types(self) -> dict[str, str]:
        """Return a dictionary of input column names and types needed for processing. This is used for schema
        validation and should be overridden by subclasses."""
        return {key: ColumnInfo.convert_pandas_dtype(str) for key in self.input_columns}

    def _process_column(self, df: pd.DataFrame) -> pd.Series:
        """
        Concatenate the values from the input columns and return the result as a Series.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the columns.

        Returns
        -------
        pandas.Series
            The processed column as a string Series.
        """

        first_col = df[self.input_columns[0]].astype(str)

        return first_col.str.cat(others=df[self.input_columns[1:]].astype(str), sep=self.sep)


@dataclasses.dataclass
class IncrementColumn(DateTimeColumn):
    """
    Subclass of `DateTimeColumn`, counts the unique occurrences of a value in `groupby_column` over a specific time
    window `period` based on dates in the `input_name` field.

    Attributes
    ----------
    groupby_column : str
        The column to group by.
    period : str
        The period to use when grouping.

    Methods
    -------
    _process_column(df: pandas.DataFrame) -> pandas.Series
        Count the unique occurrences and return the result as a Series.

    """

    groupby_column: str
    period: str = "D"

    def get_input_column_types(self) -> dict[str, str]:
        """
        Return a dictionary of input column names and types needed for processing. This is used for schema
        validation and should be overridden by subclasses.
        """
        return {
            self.input_name: ColumnInfo.convert_pandas_dtype(datetime),
            self.groupby_column: ColumnInfo.convert_pandas_dtype(str)
        }

    def _process_column(self, df: pd.DataFrame) -> pd.Series:
        """
        Count the unique occurrences and return the result as a Series.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the columns.

        Returns
        -------
        pandas.Series
            The processed column as an integer Series.
        """

        period = df[self.input_name].dt.to_period(self.period)

        # Create the `groupby_column`, per-period log count
        return df.groupby([self.groupby_column, period]).cumcount()


@dataclasses.dataclass
class DistinctIncrementColumn(RenameColumn):
    """
    Subclass of `RenameColumn`, counts the unique occurrences of a value in `groupby_column` over a specific time window
    `period` based on dates in the `timestamp_column` field. Only increments the count when the value in `input_name`
    changes.

    Attributes
    ----------
    groupby_column : str
        The column to group by.
    period : str
        The period to use when grouping.
    timestamp_column : str
        The column to use for determining the period.

    """

    groupby_column: str = "username"
    period: str = "D"
    timestamp_column: str = "timestamp"

    def get_input_column_types(self) -> dict[str, str]:
        """
        Return a dictionary of input column names and types needed for processing. This is used for schema
        validation and should be overridden by subclasses.
        """
        return {
            self.input_name: ColumnInfo.convert_pandas_dtype(str),
            self.groupby_column: ColumnInfo.convert_pandas_dtype(str),
            self.timestamp_column: ColumnInfo.convert_pandas_dtype(datetime),
        }

    def _process_column(self, df: pd.DataFrame) -> pd.Series:
        """
        Count the unique occurrences and return the result as a Series.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the columns.

        Returns
        -------
        pandas.Series
            The processed column as an integer Series.
        """

        per_period = df[self.timestamp_column].dt.to_period(self.period)

        cat_col: pd.Series = df.groupby([per_period, self.groupby_column
                                         ])[self.input_name].transform(lambda x: pd.factorize(x.fillna("nan"))[0] + 1)

        increment_col = pd.concat([cat_col, df[self.groupby_column]],
                                  axis=1).groupby([per_period, self.groupby_column
                                                   ])[self.input_name].expanding(1).max().droplevel(0).droplevel(0)

        return increment_col.astype(self.get_pandas_dtype())


@dataclasses.dataclass
class PreparedDFInfo:
    """
    Represents the result of preparing a DataFrame along with avilable columns to be preserved.

    Attributes
    ----------
    df : DataFrameType
        The prepared DataFrame.
    columns_to_preserve : list[str]
        A list of column names that are to be preserved.
    """
    df: DataFrameType
    columns_to_preserve: list[str]


def _json_flatten(df_input: DataFrameType,
                  input_columns: dict[str, str],
                  json_cols: list[str],
                  preserve_re: re.Pattern = None):
    """
    Prepares a DataFrame for processing by flattening JSON columns and converting to Pandas if necessary. Will remove
    all columns that are not specified in `input_columns` or matched by `preserve_re`.

    Parameters
    ----------
    df_input : DataFrameType
        DataFrame to process.
    input_columns : dict[str, str]
        The final input columns that are needed for processing. All other columns will be removed
    json_cols : list[str]
        List of JSON columns to flatten.
    preserve_re : re.Pattern, optional
        A RegEx where matching column names will be preserved, by default None

    Returns
    -------
    DataFrameType
        The processed DataFrame.
    """

    columns_to_preserve = set()

    if (preserve_re):
        columns_to_preserve.update(col for col in df_input.columns if re.match(preserve_re, col))

    # Early exit
    if (json_cols is None or len(json_cols) == 0):
        return PreparedDFInfo(df=df_input, columns_to_preserve=list(columns_to_preserve))

    # Check if we even have any JSON columns to flatten
    if (not df_input.columns.intersection(json_cols).empty):
        is_cudf = is_cudf_type(df_input)

        if (is_cudf):
            df_input = df_input.to_pandas()

        json_normalized = []
        columns_to_keep = list(df_input.columns)
        for col in json_cols:
            if (col not in columns_to_keep):
                continue

            pd_series = df_input[col]

            # Get the flattened columns
            pd_series = pd_series.apply(lambda x: x if isinstance(x, dict) else json.loads(x))
            pdf_norm = pd.json_normalize(pd_series)

            # Prefix column names with the JSON column name
            pdf_norm.rename(columns=lambda x, col=col: col + "." + x, inplace=True)

            # Reset the index otherwise there is a conflict
            pdf_norm.set_index(df_input.index, inplace=True)

            json_normalized.append(pdf_norm)

            if (preserve_re is None or not preserve_re.match(col)):
                columns_to_keep.remove(col)

        # Combine the original DataFrame with the normalized JSON columns
        df_input = pd.concat([df_input[columns_to_keep]] + json_normalized, axis=1)

        if (is_cudf):
            import cudf
            df_input = cudf.from_pandas(df_input).reset_index(drop=True)

    # Remove all columns that are not in the input columns list. Ensure the correct types
    df_input = df_input.reindex(columns=input_columns.keys(), fill_value=None)

    df_input = df_input.astype(input_columns)

    return PreparedDFInfo(df=df_input, columns_to_preserve=list(columns_to_preserve))


def _resolve_json_output_columns(json_cols: list[str], input_cols: dict[str, str]) -> list[tuple[str, str]]:
    """
    Resolves JSON output columns from an input schema.

    Parameters
    ----------
    input_schema : DataFrameInputSchema
        The input schema to resolve the JSON output columns from.

    Returns
    -------
    list of tuples
        A list of tuples where each tuple is a pair of column name and its data type.
    """

    json_output_candidates = list(input_cols.items())

    output_cols = []

    for col in json_output_candidates:
        cnsplit = col[0].split('.')
        if (len(cnsplit) > 1 and cnsplit[0] in json_cols):
            output_cols.append(col)

    return output_cols


@dataclasses.dataclass
class DataFrameInputSchema:
    """
    Defines the schema specifying the columns to be included in the output `DataFrame`.

    Attributes
    ----------
    json_columns : List[str]
        The columns to include in the output DataFrame.
    column_info : List[ColumnInfo]
        Information about the columns.
    preserve_columns : List[str]
        The columns to preserve.
    row_filter : Callable[[pandas.DataFrame], pandas.DataFrame]
        A function to filter the rows of the DataFrame.

    Methods
    -------
    __post_init__()
        Compile the `preserve_columns` into a regular expression.

    """

    json_columns: typing.List[str] = dataclasses.field(default_factory=list)
    column_info: typing.List[ColumnInfo] = dataclasses.field(default_factory=list)
    preserve_columns: typing.Pattern[str] = dataclasses.field(default_factory=list)
    row_filter: typing.Callable[[pd.DataFrame], pd.DataFrame] = None

    json_output_columns: typing.List[tuple[str, str]] = dataclasses.field(init=False, repr=False)
    input_columns: typing.Dict[str, str] = dataclasses.field(init=False, repr=False)
    output_columns: typing.List[tuple[str, str]] = dataclasses.field(init=False, repr=False)

    prep_dataframe: typing.Callable[[pd.DataFrame], typing.List[str]] = dataclasses.field(init=False, repr=False)

    def __post_init__(self):
        """
        Compile the `preserve_columns` into a regular expression.
        """
        input_preserve_columns = self.preserve_columns

        # Ensure preserve_columns is a list
        if (not isinstance(input_preserve_columns, list)):
            input_preserve_columns = [input_preserve_columns]

        # Compile the regex
        if (input_preserve_columns is not None and len(input_preserve_columns) > 0):
            input_preserve_columns = re.compile(f"({'|'.join(input_preserve_columns)})")
        else:
            input_preserve_columns = None

        self.preserve_columns = input_preserve_columns
        self.output_columns = []

        input_columns_dict = {}
        for col_info in self.column_info:
            self.output_columns.append((col_info.name, col_info.dtype))

            col_input_types = col_info.get_input_column_types()

            overlapping_keys = set(col_input_types.keys()).intersection(input_columns_dict.keys())

            for key in overlapping_keys:

                if (col_input_types[key] != input_columns_dict[key]):
                    raise ValueError(f"Column input '{key}' is defined with conflicting types: "
                                     f"{col_input_types[key]} != {input_columns_dict[key]}")

            # Update the dictionary with the input columns
            input_columns_dict.update(col_info.get_input_column_types())

        self.input_columns = input_columns_dict

        self.json_output_columns = _resolve_json_output_columns(self.json_columns, self.input_columns)

        self.prep_dataframe = partial(_json_flatten,
                                      input_columns=self.input_columns,
                                      json_cols=self.json_columns,
                                      preserve_re=self.preserve_columns)
