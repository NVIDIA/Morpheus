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

import filecmp
import os
import pathlib
import typing

import numpy as np
import pytest

import cudf

from _utils import TEST_DIRS
from _utils import assert_path_exists
from _utils import assert_results
from _utils.dataset_manager import DatasetManager
from morpheus.common import FileTypes
from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.io.deserializers import read_file_to_df
from morpheus.io.serializers import write_df_to_file
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.stage_decorator import stage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage


@pytest.mark.slow
@pytest.mark.gpu_and_cpu_mode
@pytest.mark.parametrize("input_type", ["csv", "jsonlines", "parquet"])
@pytest.mark.parametrize("use_pathlib", [False, True])
@pytest.mark.parametrize("output_type", ["csv", "json", "jsonlines"])
@pytest.mark.parametrize("flush", [False, True], ids=["no_flush", "flush"])
@pytest.mark.parametrize("repeat", [1, 2, 5], ids=["repeat1", "repeat2", "repeat5"])
def test_file_rw_pipe(tmp_path: pathlib.Path,
                      config: Config,
                      input_type: str,
                      use_pathlib: bool,
                      output_type: str,
                      flush: bool,
                      repeat: int):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, f'filter_probs.{input_type}')

    if use_pathlib:
        input_file = pathlib.Path(input_file)

    validation_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = os.path.join(tmp_path, f'results.{output_type}')

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file, repeat=repeat))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False, flush=flush))
    pipe.run()

    assert_path_exists(out_file)

    validation_data = np.loadtxt(validation_file, delimiter=",", skiprows=1)

    # Repeat the input data
    validation_data = np.tile(validation_data, (repeat, 1))

    if output_type == "csv":
        # The output data will contain an additional id column that we will need to slice off
        output_data = np.loadtxt(out_file, delimiter=",", skiprows=1)
        output_data = output_data[:, 1:]
    elif output_type in ("json", "jsonlines"):  # assume json
        df = read_file_to_df(out_file, file_type=FileTypes.Auto)
        output_data = df.values
    elif output_type == "parquet":
        df = read_file_to_df(out_file, file_type=FileTypes.Auto)
        output_data = df.values
    else:
        assert False, "Unknown file extension"

    # Somehow 0.7 ends up being 0.7000000000000001
    output_data = np.around(output_data, 2)
    assert output_data.tolist() == validation_data.tolist()


@pytest.mark.gpu_and_cpu_mode
def test_file_read_json(config: Config):
    src_file = os.path.join(TEST_DIRS.tests_data_dir, "simple.json")

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=src_file, parser_kwargs={"lines": False}))
    sink_stage = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    messages = sink_stage.get_messages()

    assert (len(messages) == 1)

    meta = messages[0]

    assert (len(meta.df) == 4)
    assert (len(meta.df.columns) == 3)


@pytest.mark.slow
@pytest.mark.gpu_and_cpu_mode
@pytest.mark.usefixtures("chdir_tmpdir")
def test_to_file_no_path(tmp_path: pathlib.Path, config: Config):
    """
    Test to ensure issue #48 is fixed
    """
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = "test.csv"

    assert os.path.realpath(os.curdir) == tmp_path.as_posix()

    assert not os.path.exists(tmp_path / out_file)
    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))
    pipe.run()

    assert_path_exists(tmp_path / out_file)


@pytest.mark.slow
@pytest.mark.gpu_and_cpu_mode
@pytest.mark.parametrize("input_type", ["csv", "jsonlines", "parquet"])
@pytest.mark.parametrize("output_type", ["csv", "json", "jsonlines"])
def test_file_rw_multi_segment_pipe(tmp_path: pathlib.Path, config: Config, input_type: str, output_type: str):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, f'filter_probs.{input_type}')
    validation_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = os.path.join(tmp_path, f'results.{output_type}')

    if (input_type == "parquet"):
        CppConfig.set_should_use_cpp(False)

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file))
    pipe.add_segment_boundary(MessageMeta)
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))
    pipe.run()

    assert_path_exists(out_file)

    validation_data = np.loadtxt(validation_file, delimiter=",", skiprows=1)

    if output_type == "csv":
        # The output data will contain an additional id column that we will need to slice off
        output_data = np.loadtxt(out_file, delimiter=",", skiprows=1)
        output_data = output_data[:, 1:]
    else:  # assume json
        df = read_file_to_df(out_file, file_type=FileTypes.Auto)
        output_data = df.values

    # Somehow 0.7 ends up being 0.7000000000000001
    output_data = np.around(output_data, 2)
    assert output_data.tolist() == validation_data.tolist()


@pytest.mark.slow
@pytest.mark.gpu_and_cpu_mode
@pytest.mark.parametrize("input_file",
                         [
                             os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv"),
                             os.path.join(TEST_DIRS.tests_data_dir, "filter_probs_w_id_col.csv")
                         ])
def test_file_rw_index_pipe(tmp_path: pathlib.Path, config: Config, input_file: str):
    out_file = os.path.join(tmp_path, 'results.csv')

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False, include_index_col=False))
    pipe.run()

    assert_path_exists(out_file)

    validation_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    validation_data = np.loadtxt(validation_file, delimiter=",", skiprows=1)
    output_data = np.loadtxt(out_file, delimiter=",", skiprows=1)

    # Somehow 0.7 ends up being 0.7000000000000001
    output_data = np.around(output_data, 2)
    assert output_data.tolist() == validation_data.tolist()


@pytest.mark.gpu_and_cpu_mode
@pytest.mark.parametrize("input_file,extra_kwargs",
                         [(os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv"), {
                             "include_header": True, "include_index_col": False
                         }), (os.path.join(TEST_DIRS.tests_data_dir, "filter_probs_w_id_col.csv"), {
                             "include_header": True
                         }), (os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.jsonlines"), {})],
                         ids=["CSV", "CSV_ID", "JSON"])
def test_file_roundtrip(tmp_path: pathlib.Path, input_file: str, extra_kwargs: dict[str, typing.Any]):

    # Output file should be same type as input
    out_file = os.path.join(tmp_path, f'results{os.path.splitext(input_file)[1]}')

    # Read the dataframe
    df = read_file_to_df(input_file, df_type='cudf')

    # Write the dataframe
    write_df_to_file(df=df, file_name=out_file, **extra_kwargs)

    assert_path_exists(out_file)

    assert filecmp.cmp(input_file, out_file)


@pytest.mark.parametrize("input_file",
                         [
                             os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv"),
                             os.path.join(TEST_DIRS.tests_data_dir, "filter_probs_w_id_col.csv"),
                             os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.jsonlines"),
                             os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.parquet")
                         ],
                         ids=["CSV", "CSV_ID", "JSON", "PARQUET"])
def test_read_cpp_compare(input_file: str):

    # First read with python
    CppConfig.set_should_use_cpp(False)
    df_python = read_file_to_df(input_file, df_type='cudf')

    # Now with C++
    CppConfig.set_should_use_cpp(True)
    df_cpp = read_file_to_df(input_file, df_type='cudf')

    DatasetManager.assert_df_equal(df_python, df_cpp)


@pytest.mark.slow
@pytest.mark.gpu_and_cpu_mode
@pytest.mark.parametrize("output_type", ["csv", "json", "jsonlines"])
def test_file_rw_serialize_deserialize_pipe(tmp_path: pathlib.Path, config: Config, output_type: str):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = os.path.join(tmp_path, f'results.{output_type}')

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file, iterative=False))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(SerializeStage(config))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))
    pipe.run()

    assert_path_exists(out_file)

    input_data = np.loadtxt(input_file, delimiter=",", skiprows=1)

    if output_type == "csv":
        # The output data will contain an additional id column that we will need to slice off
        output_data = np.loadtxt(out_file, delimiter=",", skiprows=1)
        output_data = output_data[:, 1:]
    else:  # assume json
        df = read_file_to_df(out_file, file_type=FileTypes.Auto)
        output_data = df.values

    # Somehow 0.7 ends up being 0.7000000000000001
    output_data = np.around(output_data, 2)
    assert output_data.tolist() == input_data.tolist()


@pytest.mark.slow
@pytest.mark.parametrize("output_type", ["csv", "json", "jsonlines"])
def test_file_rw_serialize_deserialize_multi_segment_pipe(tmp_path: pathlib.Path, config: Config, output_type: str):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = os.path.join(tmp_path, f'results.{output_type}')

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file, iterative=False))
    pipe.add_segment_boundary(MessageMeta)
    pipe.add_stage(DeserializeStage(config))
    pipe.add_segment_boundary(ControlMessage)
    pipe.add_stage(SerializeStage(config))
    pipe.add_segment_boundary(MessageMeta)
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))
    pipe.run()

    assert_path_exists(out_file)

    input_data = np.loadtxt(input_file, delimiter=",", skiprows=1)

    if output_type == "csv":
        # The output data will contain an additional id column that we will need to slice off
        output_data = np.loadtxt(out_file, delimiter=",", skiprows=1)
        output_data = output_data[:, 1:]
    else:  # assume json
        df = read_file_to_df(out_file, file_type=FileTypes.Auto)
        output_data = df.values

    # Somehow 0.7 ends up being 0.7000000000000001
    output_data = np.around(output_data, 2)
    assert output_data.tolist() == input_data.tolist()


@pytest.mark.slow
@pytest.mark.parametrize("use_get_set_data", [False, True])
def test_sliced_meta_nulls(config: Config, use_get_set_data: bool):
    """
    Test reproduces Morpheus issue #2011
    Issue occurrs when the length of the dataframe is larger than the pipeline batch size
    """
    config.pipeline_batch_size = 256

    input_df = cudf.DataFrame({"a": range(1024)})
    expected_df = cudf.DataFrame({"a": range(1024), "copy": range(1024)})

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[input_df]))
    pipe.add_stage(DeserializeStage(config))

    @stage(execution_modes=(config.execution_mode, ), needed_columns={"copy": TypeId.INT64})
    def copy_col(msg: ControlMessage) -> ControlMessage:
        meta = msg.payload()

        if use_get_set_data:
            a_col = meta.get_data('a')
            assert len(a_col) <= config.pipeline_batch_size
            meta.set_data("copy", a_col)
        else:
            with meta.mutable_dataframe() as df:
                assert len(df) <= config.pipeline_batch_size
                df['copy'] = df['a']

        return msg

    pipe.add_stage(copy_col(config))
    pipe.add_stage(SerializeStage(config))
    cmp_stage = pipe.add_stage(CompareDataFrameStage(config, compare_df=expected_df))
    pipe.run()

    assert_results(cmp_stage.get_results())
