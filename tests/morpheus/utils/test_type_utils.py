#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import types
import typing

import pandas as pd
import pytest

import cudf

from morpheus.config import ExecutionMode
from morpheus.utils.type_aliases import DataFrameModule
from morpheus.utils.type_utils import df_type_str_to_exec_mode
from morpheus.utils.type_utils import df_type_str_to_pkg
from morpheus.utils.type_utils import exec_mode_to_df_type_str
from morpheus.utils.type_utils import get_df_class
from morpheus.utils.type_utils import get_df_pkg
from morpheus.utils.type_utils import is_cudf_type


@pytest.mark.parametrize("mode, expected",
                         [(ExecutionMode.GPU, cudf.DataFrame), (ExecutionMode.CPU, pd.DataFrame),
                          ("cudf", cudf.DataFrame), ("pandas", pd.DataFrame)])
def test_get_df_class(mode: typing.Union[ExecutionMode, DataFrameModule], expected: types.ModuleType):
    assert get_df_class(mode) is expected


@pytest.mark.parametrize("mode, expected", [(ExecutionMode.GPU, cudf), (ExecutionMode.CPU, pd), ("cudf", cudf),
                                            ("pandas", pd)])
def test_get_df_pkg(mode: typing.Union[ExecutionMode, DataFrameModule], expected: types.ModuleType):
    assert get_df_pkg(mode) is expected


@pytest.mark.parametrize(
    "obj, expected",
    [
        (cudf.DataFrame(), True),
        (cudf.Series(), True),
        (cudf.Index([]), True),
        (cudf.RangeIndex(0), True),
        (pd.DataFrame(), False),
        (pd.Series(), False),
        (pd.Index([]), False),
        (pd.RangeIndex(0), False),
        (None, False),
        (0, False),
        ("test", False),
    ],
    ids=[
        "cudf.DataFrame",
        "cudf.Series",
        "cudf.Index",
        "cudf.RangeIndex",
        "pd.DataFrame",
        "pd.Series",
        "pd.Index",
        "pd.RangeIndex",
        "None",
        "int",
        "str"
    ],
)
def test_is_cudf_type(obj: typing.Any, expected: bool):
    assert is_cudf_type(obj) == expected


@pytest.mark.parametrize("df_type_str, expected", [("cudf", cudf), ("pandas", pd)], ids=["cudf", "pandas"])
def test_df_type_str_to_pkg(df_type_str: DataFrameModule, expected: types.ModuleType):
    assert df_type_str_to_pkg(df_type_str) is expected


@pytest.mark.parametrize("invalid_type_str", ["invalid", "cuDF", "Pandas"])
def test_df_type_str_to_pkg_invalid(invalid_type_str: typing.Any):
    with pytest.raises(ValueError, match="Invalid DataFrame type string"):
        df_type_str_to_pkg(invalid_type_str)


@pytest.mark.parametrize("df_type_str, expected", [("cudf", ExecutionMode.GPU), ("pandas", ExecutionMode.CPU)],
                         ids=["cudf", "pandas"])
def test_df_type_str_to_exec_mode(df_type_str: DataFrameModule, expected: ExecutionMode):
    assert df_type_str_to_exec_mode(df_type_str) == expected


@pytest.mark.parametrize("invalid_type_str", ["invalid", "cuDF", "Pandas"])
def test_df_type_str_to_exec_mode_invalid(invalid_type_str: typing.Any):
    with pytest.raises(ValueError, match="Invalid DataFrame type string"):
        df_type_str_to_exec_mode(invalid_type_str)


@pytest.mark.parametrize("exec_mode, expected", [(ExecutionMode.GPU, "cudf"), (ExecutionMode.CPU, "pandas")],
                         ids=["GPU", "CPU"])
def test_exec_mode_to_df_type_str(exec_mode: ExecutionMode, expected: DataFrameModule):
    assert exec_mode_to_df_type_str(exec_mode) == expected
