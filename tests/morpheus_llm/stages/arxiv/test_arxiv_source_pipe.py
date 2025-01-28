#!/usr/bin/env python
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

import os
import shutil
from unittest import mock

import cudf

from _utils import assert_results
from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.arxiv_source import ArxivSource
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage


def test_arxiv_source_pipeline(mock_arxiv_search: mock.MagicMock, config: Config, tmp_path: str, pdf_file: str):
    # just take the first PDF file
    mock_arxiv_search.results.return_value = [mock_arxiv_search.results.return_value[0]]

    # Pre populate the cache dir with a PDF file
    cached_pdf = os.path.join(tmp_path, "apples.pdf")
    shutil.copyfile(pdf_file, cached_pdf)

    content = "Morpheus\nunittest"
    page_content_col = [content]
    source_col = [cached_pdf]
    page_col = [0]
    type_col = ["Document"]

    expected_df = cudf.DataFrame({
        "page_content": page_content_col, "source": source_col, "page": page_col, "type": type_col
    })

    # The ArxivSource sets a pe_count of 6 for the process_pages node, and we need at least that number of threads
    # in the config to run the pipeline
    config.num_threads = 6
    pipe = LinearPipeline(config)
    pipe.set_source(ArxivSource(config, query="unittest", cache_dir=tmp_path))
    comp_stage = pipe.add_stage(CompareDataFrameStage(config, expected_df))
    pipe.run()

    assert_results(comp_stage.get_results())

    mock_arxiv_search.assert_called_once()
    mock_arxiv_search.results.assert_called_once()
    mock_results: list[mock.MagicMock] = mock_arxiv_search.results.return_value
    for mock_result in mock_results:
        # Since we pre-populated the cache dir, we should not have attempted to download any pdfs
        mock_result.download_pdf.assert_not_called()
