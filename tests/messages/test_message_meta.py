#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    use_cpp: bool,  # pylint: disable=unused-argument
    dataset: DatasetManager,
    index_type: typing.Literal['normal', 'skip', 'dup', 'down',
                               'updown']) -> typing.Union[cudf.DataFrame, pd.DataFrame]:
    filter_probs_df = dataset["filter_probs.csv"]

    if (index_type == "normal"):
        return filter_probs_df

    if (index_type == "skip"):
        # Skip some rows
        return filter_probs_df.iloc[::3, :].copy()

    if (index_type == "dup"):
        # Duplicate
        return dataset.dup_index(filter_probs_df, count=2)

    if (index_type == "down"):
        # Reverse
        return filter_probs_df.iloc[::-1, :].copy()

    if (index_type == "updown"):
        # Go up then down
        down = filter_probs_df.iloc[::-1, :].copy()

        # Increase the index to keep them unique
        down.index += len(down)

        if isinstance(filter_probs_df, pd.DataFrame):
            concat_fn = pd.concat
        else:
            concat_fn = cudf.concat

        out_df = concat_fn([filter_probs_df, down])

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


def test_mutable_dataframe(df: DataFrameType):

    meta = MessageMeta(df)

    with meta.mutable_dataframe() as df_:
        df_['v2'].iloc[3] = 47

    assert meta.copy_dataframe()['v2'].iloc[3] == 47


def test_using_ctx_outside_with_block(df: DataFrameType):

    meta = MessageMeta(df)

    ctx = meta.mutable_dataframe()

    # ctx.fillna() & ctx.col
    pytest.raises(AttributeError, getattr, ctx, 'fillna')

    # ctx[col]
    pytest.raises(AttributeError, operator.getitem, ctx, 'col')

    # ctx[col] = 5
    pytest.raises(AttributeError, operator.setitem, ctx, 'col', 5)


def test_copy_dataframe(df: DataFrameType):

    meta = MessageMeta(df)

    copied_df = meta.copy_dataframe()

    DatasetManager.assert_df_equal(copied_df, df, assert_msg="Should be identical")
    assert copied_df is not df, "But should be different instances"

    # Try setting a single value on the copy
    cdf = meta.copy_dataframe()
    cdf['v2'].iloc[3] = 47
    DatasetManager.assert_df_equal(meta.copy_dataframe(), df, assert_msg="Should be identical")


@pytest.mark.use_cpp
def test_pandas_df_cpp(dataset_pandas: DatasetManager):
    """
    Test for issue #821, calling the `df` property returns an empty cudf dataframe.
    """
    df = dataset_pandas["filter_probs.csv"]
    assert isinstance(df, pd.DataFrame)

    meta = MessageMeta(df)
    assert isinstance(meta, MessageMetaCpp)
    assert isinstance(meta.df, cudf.DataFrame)
    DatasetManager.assert_compare_df(meta.df, df)


def test_cast(config: Config, dataset: DatasetManager):  # pylint: disable=unused-argument
    """
    Test tcopy constructor
    """
    df = dataset["filter_probs.csv"]
    meta1 = MessageMeta(df)

    meta2 = MessageMeta(meta1)
    assert isinstance(meta2, MessageMeta)

    DatasetManager.assert_compare_df(meta2.copy_dataframe(), df)


@pytest.mark.use_pandas
@pytest.mark.use_python
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
@pytest.mark.use_python
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
