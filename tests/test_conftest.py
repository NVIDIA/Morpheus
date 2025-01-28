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

import typing

import pandas as pd
import pytest

import cudf

from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.config import ExecutionMode
from morpheus.utils.type_aliases import DataFrameModule
from morpheus.utils.type_utils import exec_mode_to_df_type_str


def exec_mode_to_cpp_mode(exec_mode: ExecutionMode) -> bool:
    return exec_mode == ExecutionMode.GPU


@pytest.fixture(name="exec_mode_from_marker", scope="function")
def exec_mode_from_marker_fixture(request: pytest.FixtureRequest) -> ExecutionMode:

    gpu_mode = len([x for x in request.node.iter_markers("gpu_mode") if "added_by" in x.kwargs]) > 0
    cpu_mode = len([x for x in request.node.iter_markers("cpu_mode") if "added_by" in x.kwargs]) > 0

    assert gpu_mode != cpu_mode

    if gpu_mode:
        return ExecutionMode.GPU

    return ExecutionMode.CPU


@pytest.fixture(name="cpp_mode_from_marker", scope="function")
def cpp_mode_from_marker_fixture(request: pytest.FixtureRequest) -> bool:

    gpu_mode = len([x for x in request.node.iter_markers("gpu_mode") if "added_by" in x.kwargs]) > 0
    cpu_mode = len([x for x in request.node.iter_markers("cpu_mode") if "added_by" in x.kwargs]) > 0

    assert gpu_mode != cpu_mode

    return gpu_mode


@pytest.fixture(name="df_type_from_marker", scope="function")
def df_type_from_marker_fixture(request: pytest.FixtureRequest) -> bool:

    use_cudf = len([x for x in request.node.iter_markers("use_cudf") if "added_by" in x.kwargs]) > 0
    use_pandas = len([x for x in request.node.iter_markers("use_pandas") if "added_by" in x.kwargs]) > 0

    assert use_cudf != use_pandas

    return "cudf" if use_cudf else "pandas"


@pytest.mark.use_cudf
@pytest.mark.use_pandas
def test_dataset_works_with_marks(dataset: DatasetManager):
    # Test is parameterized so df runs twice, once as pandas and another time as cudf
    df = dataset["filter_probs.csv"]
    assert isinstance(df, (pd.DataFrame, cudf.DataFrame))


def test_dataset_only_pandas(dataset_pandas: DatasetManager):
    # Test only runs with pandas
    df = dataset_pandas["filter_probs.csv"]
    assert isinstance(df, pd.DataFrame)


def test_dataset_only_cudf(dataset_cudf: DatasetManager):
    # Test only runs with cudf
    df = dataset_cudf["filter_probs.csv"]
    assert isinstance(df, cudf.DataFrame)


def test_dataset_both(dataset: DatasetManager):
    # By default, requesting dataset will parameterize both
    df = dataset["filter_probs.csv"]
    assert isinstance(df, (pd.DataFrame, cudf.DataFrame))


def test_dataset_manager_singleton(df_type: typing.Literal["cudf", "pandas"]):
    dataset = DatasetManager(df_type=df_type)
    assert dataset.default_df_type == df_type
    assert getattr(dataset, df_type) is dataset
    assert DatasetManager(df_type=df_type) is dataset

    alt_type = DatasetManager.get_alt_df_type(df_type=df_type)
    assert df_type != alt_type
    assert DatasetManager(alt_type) is not dataset
    assert getattr(dataset, alt_type) is not dataset


def test_dataset_dftype(dataset: DatasetManager):
    df = dataset["filter_probs.csv"]  # type will match the df_type parameter

    if dataset.default_df_type == 'pandas':
        assert isinstance(df, pd.DataFrame)
    else:
        assert isinstance(df, cudf.DataFrame)


def test_dataset_properties(dataset: DatasetManager):
    pdf = dataset.pandas["filter_probs.csv"]
    assert isinstance(pdf, pd.DataFrame)

    cdf = dataset.cudf["filter_probs.csv"]
    assert isinstance(cdf, cudf.DataFrame)


def test_dataset_reader_args(dataset: DatasetManager):
    # When `filter_nulls=False` this returns all 20 rows, or 3 when True
    input_file = 'examples/abp_pcap_detection/abp_pcap.jsonlines'
    assert len(dataset.get_df(input_file, filter_nulls=False)) == 20
    assert len(dataset[input_file]) == 3

    # input_file is now in the cache with the default args, double check to make sure we still ignore the cache
    assert len(dataset.get_df(input_file, filter_nulls=False)) == 20


# === No Marks ===
def test_no_mark():
    assert CppConfig.get_should_use_cpp()


# === No Marks ===
@pytest.mark.gpu_mode
def test_mark_gpu_mode():
    assert CppConfig.get_should_use_cpp()


@pytest.mark.cpu_mode
def test_mark_cpu_mode():
    assert not CppConfig.get_should_use_cpp()


# === Marks and Config ===
@pytest.mark.gpu_mode
@pytest.mark.usefixtures("config")
def test_mark_and_config_gpu_mode():
    assert CppConfig.get_should_use_cpp()


@pytest.mark.cpu_mode
def test_mark_and_config_cpu_mode(config: Config):
    assert not CppConfig.get_should_use_cpp()
    assert config.execution_mode == ExecutionMode.CPU


@pytest.mark.gpu_and_cpu_mode
def test_gpu_and_cpu_mode(config: Config, exec_mode_from_marker: ExecutionMode):
    assert config.execution_mode == exec_mode_from_marker


def test_mark_and_config_neither(config: Config):
    assert CppConfig.get_should_use_cpp()
    assert config.execution_mode == ExecutionMode.GPU


# === Fixture ===
@pytest.mark.gpu_mode
def test_fixture_gpu_mode(execution_mode: ExecutionMode):
    assert execution_mode == ExecutionMode.GPU
    assert CppConfig.get_should_use_cpp()


@pytest.mark.cpu_mode
def test_fixture_cpu_mode(execution_mode: ExecutionMode):
    assert execution_mode == ExecutionMode.CPU
    assert not CppConfig.get_should_use_cpp()


def test_fixture_neither(execution_mode: ExecutionMode):
    assert execution_mode == ExecutionMode.GPU
    assert CppConfig.get_should_use_cpp()


# === Config Fixture ===
@pytest.mark.usefixtures("config")
def test_config_fixture():
    assert CppConfig.get_should_use_cpp()


class TestNoMarkerClass:

    def test_no_marker(self):
        assert CppConfig.get_should_use_cpp()

    @pytest.mark.cpu_mode
    def test_python_marker(self, execution_mode: ExecutionMode):
        assert execution_mode == ExecutionMode.CPU
        assert not CppConfig.get_should_use_cpp()

    @pytest.mark.gpu_mode
    def test_cpp_marker(self, execution_mode: ExecutionMode):
        assert execution_mode == ExecutionMode.GPU
        assert CppConfig.get_should_use_cpp()

    @pytest.mark.slow
    def test_other_marker(self, execution_mode: ExecutionMode):
        assert execution_mode == ExecutionMode.GPU
        assert CppConfig.get_should_use_cpp()


@pytest.mark.cpu_mode
class TestPythonMarkerClass:

    def test_no_marker(self):
        assert not CppConfig.get_should_use_cpp()

    def test_with_fixture(self, execution_mode: ExecutionMode):
        assert execution_mode == ExecutionMode.CPU
        assert not CppConfig.get_should_use_cpp()

    @pytest.mark.cpu_mode
    def test_extra_marker(self, execution_mode: ExecutionMode):
        assert execution_mode == ExecutionMode.CPU
        assert not CppConfig.get_should_use_cpp()


@pytest.mark.gpu_mode
class TestCppMarkerClass:

    def test_no_marker(self):
        assert CppConfig.get_should_use_cpp()

    def test_with_fixture(self, execution_mode: ExecutionMode):
        assert execution_mode == ExecutionMode.GPU
        assert CppConfig.get_should_use_cpp()

    @pytest.mark.gpu_mode
    def test_extra_marker(self):
        assert CppConfig.get_should_use_cpp()


# === DF Type ===
def test_df_type_no_marks(df_type, df_type_from_marker):
    assert df_type == df_type_from_marker


def test_df_type_matches_execution_mode(df_type: DataFrameModule, execution_mode: ExecutionMode):
    assert df_type == exec_mode_to_df_type_str(execution_mode)


@pytest.mark.use_pandas
def test_df_type_pandas_marker(df_type):
    assert df_type == "pandas"


@pytest.mark.use_cudf
def test_df_type_cudf_marker(df_type):
    assert df_type == "cudf"


@pytest.mark.use_cudf
@pytest.mark.use_pandas
def test_df_type_both_markers(df_type, df_type_from_marker):
    assert df_type == df_type_from_marker
