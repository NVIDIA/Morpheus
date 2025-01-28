#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import operator
import typing

import pandas as pd
import pytest

import cudf

from _utils.dataset_manager import DatasetManager
# pylint: disable=morpheus-incorrect-lib-from-import
from morpheus._lib.messages import MessageMeta as MessageMetaCpp
from morpheus.config import Config
from morpheus.messages.message_meta import MessageMeta
from morpheus.utils.type_aliases import DataFrameType


@pytest.fixture(name="index_type", scope="function", params=["normal", "skip", "dup", "down", "updown"])
def fixture_index_type(request: pytest.FixtureRequest) -> typing.Literal["normal", "skip", "dup", "down", "updown"]:
    return request.param


@pytest.fixture(name="df", scope="function")
def fixture_df(
    dataset: DatasetManager, index_type: typing.Literal['normal', 'skip', 'dup', 'down',
                                                        'updown']) -> typing.Union[cudf.DataFrame, pd.DataFrame]:
    test_df = dataset["test_dataframe.jsonlines"]

    if (index_type == "normal"):
        return test_df

    if (index_type == "skip"):
        # Skip some rows
        return test_df.iloc[::3, :].copy()

    if (index_type == "dup"):
        # Duplicate
        return dataset.dup_index(test_df, count=2)

    if (index_type == "down"):
        # Reverse
        return test_df.iloc[::-1, :].copy()

    if (index_type == "updown"):
        # Go up then down
        down = test_df.iloc[::-1, :].copy()

        # Increase the index to keep them unique
        down.index += len(down)

        if isinstance(test_df, pd.DataFrame):
            concat_fn = pd.concat
        else:
            concat_fn = cudf.concat

        out_df = concat_fn([test_df, down])

        assert out_df.index.is_unique

        return out_df

    assert False, "Unknown index type"


@pytest.fixture(name="is_sliceable", scope="function")
def fixture_is_sliceable(index_type: typing.Literal['normal', 'skip', 'dup', 'down', 'updown']):

    return index_type not in ("dup", "updown")


def test_count(df: DataFrameType):

    meta = MessageMeta(df)

    assert meta.count == len(df)


def test_has_sliceable_index(df: DataFrameType, is_sliceable: bool):

    meta = MessageMeta(df)
    assert meta.has_sliceable_index() == is_sliceable


def test_ensure_sliceable_index(df: DataFrameType, is_sliceable: bool):

    meta = MessageMeta(df)

    old_index_name = meta.ensure_sliceable_index()

    assert meta.has_sliceable_index()
    assert old_index_name == (None if is_sliceable else "_index_")


def test_using_ctx_outside_with_block(df: DataFrameType):

    meta = MessageMeta(df)

    ctx = meta.mutable_dataframe()

    # ctx.fillna() & ctx.col
    pytest.raises(AttributeError, getattr, ctx, 'fillna')

    # ctx[col]
    pytest.raises(AttributeError, operator.getitem, ctx, 'col')

    # ctx[col] = 5
    pytest.raises(AttributeError, operator.setitem, ctx, 'col', 5)


@pytest.mark.use_cudf
def test_update_dataframe_via_mutable_dataframe(df: DataFrameType):
    """
    Change the DF in various ways pass to cpp, read back and check if
    the updates present
    """
    meta = MessageMeta(df)
    cdf = meta.copy_dataframe()
    DatasetManager.assert_df_equal(cdf, df, assert_msg="Should be identical")
    assert cdf is not df, "But should be different instances"

    row_count = df.shape[0]

    # new struct column
    col_new_name = "bestsellers"
    col_new_struct = [{
        "book": "A Tale of Two Cities", "year": 1859
    }, {
        "book": "The Lord of the Rings", "year": 1954
    }, {
        "book": "The Little Prince", "year": 1943
    }, {
        "book": "The Hobbit", "year": 1937
    }, {
        "book": "And Then There Were None", "year": 1939
    }, {
        "book": "Dream of the Red Chamber", "year": 1791
    }, {
        "book": "The Lion, the Witch and the Wardrobe", "year": 1950
    }, {
        "book": "She: A History of Adventure", "year": 1887
    }, {
        "book": "Le Petit Larousse", "year": 1905
    }, {
        "book": "Harry Potter and the Philosopher's Stone", "year": 1997
    }]

    # if row_count is more than 10 just replicate the struct column
    if row_count > len(col_new_struct):
        col_new_struct = col_new_struct * (row_count // len(col_new_struct) + 1)

    # slice col_new_struct to match the row count
    col_new_struct = col_new_struct[:row_count]

    # add a struct column to the DF and pass it to cpp
    with meta.mutable_dataframe() as df_:
        df_.insert(0, col_new_name, col_new_struct)
    cdf = meta.copy_dataframe()
    assert col_new_name in cdf.columns
    assert cdf[col_new_name].isin(col_new_struct).all()

    # Ensure that the data matches the original data
    DatasetManager.assert_df_equal(cdf[col_new_name], col_new_struct, assert_msg="Should be identical")

    # new int column in range 1-row_count
    col_new_int = list(range(1, row_count + 1))

    # replace the struct column with int column
    with meta.mutable_dataframe() as df_:
        df_[col_new_name] = col_new_int
    cdf = meta.copy_dataframe()
    assert cdf[col_new_name].isin(col_new_int).all()

    # replace the int column back with struct column
    with meta.mutable_dataframe() as df_:
        df_[col_new_name] = col_new_struct
    cdf = meta.copy_dataframe()
    assert cdf[col_new_name].isin(col_new_struct).all()

    # Update the struct column with a new value
    new_struct = {"book": "The Great Gatsby", "year": 1925}

    # save the contents of the struct cell
    old_struct = cdf[col_new_name].iloc[0]

    # change the contents of a struct cell
    with meta.mutable_dataframe() as df_:
        df_[col_new_name].iloc[0] = new_struct
    cdf = meta.copy_dataframe()
    assert cdf[col_new_name].iloc[0] == new_struct

    # Update just the year of the struct column
    new_year = 2022
    with meta.mutable_dataframe() as df_:
        df_[col_new_name].struct.field("year").iloc[0] = new_year
    cdf = meta.copy_dataframe()
    assert cdf[col_new_name].iloc[0] == {**new_struct, **{"year": new_year}}

    # restore the contents of the struct cell
    with meta.mutable_dataframe() as df_:
        df_[col_new_name].iloc[0] = old_struct
    cdf = meta.copy_dataframe()
    assert cdf[col_new_name].iloc[0] == old_struct

    # delete the new column from the DF
    with meta.mutable_dataframe() as df_:
        df_.drop(col_new_name, axis=1, inplace=True)
    cdf = meta.copy_dataframe()
    assert col_new_name not in cdf.columns


def test_update_dataframe(df: DataFrameType):
    """
    Change the DF in various ways via cpp, read back and check if
    the updates present
    """
    meta = MessageMeta(df)
    cdf = meta.copy_dataframe()
    DatasetManager.assert_df_equal(cdf, df, assert_msg="Should be identical")
    assert cdf is not df, "But should be different instances"

    row_count = df.shape[0]

    # new struct column
    col_new_name = "bestsellers"
    col_new_struct = [{
        "book": "A Tale of Two Cities", "year": 1859
    }, {
        "book": "The Lord of the Rings", "year": 1954
    }, {
        "book": "The Little Prince", "year": 1943
    }, {
        "book": "The Hobbit", "year": 1937
    }, {
        "book": "And Then There Were None", "year": 1939
    }, {
        "book": "Dream of the Red Chamber", "year": 1791
    }, {
        "book": "The Lion, the Witch and the Wardrobe", "year": 1950
    }, {
        "book": "She: A History of Adventure", "year": 1887
    }, {
        "book": "Le Petit Larousse", "year": 1905
    }, {
        "book": "Harry Potter and the Philosopher's Stone", "year": 1997
    }]

    # if row_count is more than 10 just replicate the struct column
    if row_count > len(col_new_struct):
        col_new_struct = col_new_struct * (row_count // len(col_new_struct) + 1)

    # slice col_new_struct to match the row count
    col_new_struct = col_new_struct[:row_count]
    # add a struct column in cpp
    meta.set_data(col_new_name, col_new_struct)
    assert col_new_name in meta.get_column_names()
    assert meta.get_data()[col_new_name].isin(col_new_struct).all()  # pylint: disable=unsubscriptable-object

    # swap the contents of the first and last books
    first_book = col_new_struct[0]
    last_book = col_new_struct[-1]
    col_new_struct[0] = last_book
    col_new_struct[-1] = first_book
    meta.set_data(col_new_name, col_new_struct)
    assert col_new_name in meta.get_column_names()
    assert meta.get_data()[col_new_name].isin(col_new_struct).all()  # pylint: disable=unsubscriptable-object

    # new int column in range 1-row_count
    col_new_int_name = "col_new_int"
    col_new_int = list(range(1, row_count + 1))
    # add new int column in cpp
    meta.set_data(col_new_int_name, col_new_int)
    assert col_new_name in meta.get_column_names()
    assert meta.get_data()[col_new_int_name].isin(col_new_int).all()  # pylint: disable=unsubscriptable-object

    # multiply values in col_new_int by 2
    col_new_int = [x * 2 for x in col_new_int]
    # update new int column in cpp
    meta.set_data(col_new_int_name, col_new_int)
    assert meta.get_data()[col_new_int_name].isin(col_new_int).all()  # pylint: disable=unsubscriptable-object


@pytest.mark.gpu_mode
def test_pandas_df_cpp(dataset_pandas: DatasetManager):
    """
    Test for issue #821, calling the `df` property returns an empty cudf dataframe.
    """
    df = dataset_pandas["test_dataframe.jsonlines"]
    assert isinstance(df, pd.DataFrame)

    meta = MessageMeta(df)
    assert isinstance(meta, MessageMetaCpp)
    assert isinstance(meta.df, cudf.DataFrame)
    DatasetManager.assert_df_equal(meta.df, df, assert_msg="Should be identical")


def test_cast(config: Config, dataset: DatasetManager):  # pylint: disable=unused-argument
    """
    Test tcopy constructor
    """
    df = dataset["test_dataframe.jsonlines"]
    meta1 = MessageMeta(df)

    meta2 = MessageMeta(meta1)
    assert isinstance(meta2, MessageMeta)

    DatasetManager.assert_df_equal(meta2.copy_dataframe(), df, assert_msg="Should be identical")


@pytest.mark.use_pandas
@pytest.mark.cpu_mode
def test_cast_python_to_cpp(dataset: DatasetManager):
    """
    Test that we can cast a python MessageMeta to a C++ MessageMeta
    """
    df = dataset["filter_probs.csv"]

    py_meta = MessageMeta(df)
    assert isinstance(py_meta, MessageMeta)
    assert not isinstance(py_meta, MessageMetaCpp)

    cpp_meta = MessageMetaCpp(py_meta)
    assert isinstance(cpp_meta, MessageMeta)
    assert isinstance(cpp_meta, MessageMetaCpp)

    DatasetManager.assert_compare_df(cpp_meta.copy_dataframe(), df)


@pytest.mark.use_pandas
@pytest.mark.cpu_mode
def test_cast_cpp_to_python(dataset: DatasetManager):
    """
    Test that we can cast a a C++ MessageMeta to a python MessageMeta
    """
    df = dataset["filter_probs.csv"]
    cpp_meta = MessageMetaCpp(df)

    py_meta = MessageMeta(cpp_meta)
    assert isinstance(py_meta, MessageMeta)
    assert not isinstance(py_meta, MessageMetaCpp)

    DatasetManager.assert_compare_df(py_meta.copy_dataframe(), df)


def test_get_column_names(df: DataFrameType):
    """
    Test that we can get the column names from a MessageMeta
    """
    expected_columns = sorted(df.columns.to_list())
    meta = MessageMeta(df)

    assert sorted(meta.get_column_names()) == expected_columns


def test_cpp_meta_slicing(dataset_cudf: DatasetManager):
    """
    Test copy_range() and get_slice() of MessageMetaCpp
    """
    df = dataset_cudf["filter_probs.csv"]

    cpp_meta = MessageMetaCpp(df)
    ranges = [(0, 1), (3, 6)]
    copy_range_cpp_meta = cpp_meta.copy_ranges(ranges)
    expected_copy_range_df = cudf.concat([df[start:stop] for start, stop in ranges])
    DatasetManager.assert_compare_df(copy_range_cpp_meta.df, expected_copy_range_df)

    slice_idx = [2, 4]
    sliced_cpp_meta = cpp_meta.get_slice(slice_idx[0], slice_idx[1])
    expected_sliced_df = df[slice_idx[0]:slice_idx[1]]
    DatasetManager.assert_compare_df(sliced_cpp_meta.df, expected_sliced_df)
