#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from unittest import mock

import fsspec
import pytest

import cudf

from _utils import TEST_DIRS
from morpheus.common import FileTypes
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline.pipeline import Pipeline
from morpheus.stages.input.file_source import FileSource
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage


@pytest.mark.use_python
@pytest.mark.parametrize("input_files,watch, protocols",
                         [(["file1.json", "file2.json"], False, ["file"]),
                          (["file://file1.json", "file2.json"], False, ["file"]),
                          (["file:///file1.json"], False, ["file"]), (["test_data/*.json"], True, ["file"]),
                          (["s3://test_data/file1.json", "s3://test_data/file2.json"], False, ["s3"]),
                          (["s3://test_data/*.json"], True, ["s3"])])
def test_constructor(config, input_files, watch, protocols):
    source = FileSource(config, files=input_files, watch=watch)
    assert sorted(source._protocols) == protocols


@pytest.mark.use_python
@pytest.mark.parametrize(
    "input_files,watch,error_msg",
    [(["file1.json", "file2.json"], True, "When 'watch' is True, the 'files' should contain exactly one file path."),
     ([], True, "The 'files' cannot be empty."), ([], False, "The 'files' cannot be empty."),
     (None, True, "The 'files' cannot be empty."), (None, False, "The 'files' cannot be empty."),
     (["file1.json", "s3://test_data/file2.json"],
      True,
      "When 'watch' is True, the 'files' should contain exactly one file path."),
     (["file1.json", "s3://test_data/file2.json"],
      False,
      "Accepts same protocol input files, but it received multiple protocols.")])
def test_constructor_error(config, input_files, watch, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        FileSource(config, files=input_files, watch=watch)


@pytest.mark.use_python
@pytest.mark.parametrize("input_file,filetypes,parser_kwargs,expected_df_count",
                         [("filter_probs.json", FileTypes.Auto, {
                             "lines": False
                         }, 20), ("filter_probs.jsonlines", FileTypes.JSON, {
                             "lines": True
                         }, 20)])
def test_generate_frames(input_file, filetypes, parser_kwargs, expected_df_count):
    in_file = fsspec.open(os.path.join(TEST_DIRS.tests_data_dir, input_file))

    meta = FileSource.generate_frames(file=in_file, file_type=filetypes, parser_kwargs=parser_kwargs)

    assert len(meta.df.columns) == 4
    assert len(meta.df) == expected_df_count
    assert isinstance(meta, MessageMeta)
    assert isinstance(meta.df, cudf.DataFrame)


@pytest.mark.use_python
@pytest.mark.parametrize(
    "input_files,parser_kwargs,max_files,watch,storage_connection_kwargs,expected_result",
    [([
        "s3://rapidsai-data/cyber/morpheus/dfp/duo/DUO_2022-08-01T00_05_06.806Z.json",
        "s3://rapidsai-data/cyber/morpheus/dfp/duo/DUO_2022-08-01T12_09_47.901Z.json"
    ], {
        "lines": False, "orient": "records"
    },
      -1,
      False,
      None,
      2),
     ([
         "/rapidsai-data/cyber/morpheus/dfp/duo/DUO_2022-08-01T00_05_06.806Z.json",
         "/rapidsai-data/cyber/morpheus/dfp/duo/DUO_2022-08-01T12_09_47.901Z.json"
     ], {
         "lines": False, "orient": "records"
     },
      -1,
      False, {
          "protocol": "s3"
      },
      2), ([os.path.join(TEST_DIRS.tests_data_dir, "triton_*.csv")], None, -1, False, None, 3),
     ([f'file:/{os.path.join(TEST_DIRS.tests_data_dir, "triton_*.csv")}'], None, -1, False, None, RuntimeError),
     ([f'file:/{os.path.join(TEST_DIRS.tests_data_dir, "triton_abp_inf_results.csv")}'],
      None,
      -1,
      False,
      None,
      FileNotFoundError),
     ([f'file://{os.path.join(TEST_DIRS.tests_data_dir, "triton_*.csv")}'], None, -1, False, None, 3),
     (["s3://rapidsai-data/cyber/morpheus/dfp/duo/DUO_2022-08-01T00_05_06.806Z.json"], {
         "lines": False, "orient": "records"
     },
      1,
      True,
      None,
      1), ([os.path.join(TEST_DIRS.tests_data_dir, "triton_*.csv")], None, 2, False, None, 2),
     ([f'file://{os.path.join(TEST_DIRS.tests_data_dir, "triton_*.csv")}'], None, 3, True, None, 3)])
def test_filesource_pipe(config,
                         input_files,
                         parser_kwargs,
                         max_files,
                         watch,
                         storage_connection_kwargs,
                         expected_result):

    pipe = Pipeline(config)

    file_source_stage = FileSource(config,
                                   files=input_files,
                                   watch=watch,
                                   max_files=max_files,
                                   parser_kwargs=parser_kwargs,
                                   storage_connection_kwargs=storage_connection_kwargs)
    sink_stage = InMemorySinkStage(config)

    pipe.add_stage(file_source_stage)
    pipe.add_stage(sink_stage)

    pipe.add_edge(file_source_stage, sink_stage)

    if expected_result in (RuntimeError, FileNotFoundError):
        with pytest.raises(expected_result):
            pipe.run()
    else:
        pipe.run()

        assert len(sink_stage.get_messages()) == expected_result


@pytest.mark.use_python
@pytest.mark.parametrize("watch", [True, False])
@mock.patch.object(FileSource, '_polling_generate_frames_fsspec')
@mock.patch.object(FileSource, '_generate_frames_fsspec')
def test_build_source(mock_generate_frames_fsspec, mock_polling_generate_frames_fsspec, watch, config):
    files = ["s3://rapidsai-data/cyber/morpheus/dfp/duo/DUO_2022*.json"]
    source = FileSource(config=config, files=files, watch=watch)

    mock_node = mock.MagicMock()
    mock_builder = mock.MagicMock()
    mock_builder.make_source.return_value = mock_node
    out_stream, out_type = source._build_source(mock_builder)

    if watch:
        mock_polling_generate_frames_fsspec.assert_called_once()
        with pytest.raises(Exception):
            mock_generate_frames_fsspec.assert_called_once()
    else:
        mock_generate_frames_fsspec.assert_called_once()
        with pytest.raises(Exception):
            mock_polling_generate_frames_fsspec.assert_called_once()

    assert out_stream == mock_node
    assert out_type == fsspec.core.OpenFiles
