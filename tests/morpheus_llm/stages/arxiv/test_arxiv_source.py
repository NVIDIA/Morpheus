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
import string
import types
from unittest import mock

import pytest

import cudf

from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.stages.input.arxiv_source import ArxivSource


def test_constructor(config: Config):
    cache_dir = "/does/not/exist"
    ArxivSource(config, query="unittest", cache_dir=cache_dir, chunk_size=77, chunk_overlap=33, max_pages=100)
    assert not os.path.exists(cache_dir)


@pytest.mark.parametrize("chunk_size,chunk_overlap", [(99, 100), (100, 100)])
def test_constructor_chunk_size_error(config: Config, chunk_size: int, chunk_overlap: int):
    with pytest.raises(ValueError):
        ArxivSource(config, query="unittest", chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def _make_mock_result(file_name: str):
    result = mock.MagicMock()
    result._get_default_filename.return_value = file_name
    return result


@pytest.mark.parametrize("use_subdir", [False, True])
def test_generate_frames_cache_miss(mock_subscription: mock.MagicMock,
                                    mock_arxiv_search: mock.MagicMock,
                                    config: Config,
                                    tmp_path: str,
                                    use_subdir: bool):
    if use_subdir:
        # Tests that the cache directory is created if it doesn't exist
        cache_dir = os.path.join(tmp_path, "cache")
        assert not os.path.exists(cache_dir)
    else:
        cache_dir = tmp_path

    stage = ArxivSource(config, query="unittest", cache_dir=cache_dir)

    expected_file_paths = [os.path.join(cache_dir, "apples.pdf"), os.path.join(cache_dir, "plums.pdf")]
    assert list(stage._generate_frames(mock_subscription)) == expected_file_paths

    assert os.path.exists(cache_dir)

    mock_arxiv_search.assert_called_once()
    mock_arxiv_search.results.assert_called_once()
    mock_results: list[mock.MagicMock] = mock_arxiv_search.results.return_value
    for mock_result in mock_results:
        mock_result.download_pdf.assert_called_once()


def test_generate_frames_cache_hit(mock_subscription: mock.MagicMock,
                                   mock_arxiv_search: mock.MagicMock,
                                   config: Config,
                                   tmp_path: str):
    with open(os.path.join(tmp_path, "apples.pdf"), "w", encoding="utf-8") as f:
        f.write("apples")

    stage = ArxivSource(config, query="unittest", cache_dir=tmp_path)

    expected_file_paths = [os.path.join(tmp_path, "apples.pdf"), os.path.join(tmp_path, "plums.pdf")]
    assert list(stage._generate_frames(mock_subscription)) == expected_file_paths

    mock_arxiv_search.assert_called_once()
    mock_arxiv_search.results.assert_called_once()
    mock_results: list[mock.MagicMock] = mock_arxiv_search.results.return_value
    for mock_result in mock_results:
        if mock_result._get_default_filename.return_value == "apples.pdf":
            mock_result.download_pdf.assert_not_called()
        else:
            mock_result.download_pdf.assert_called_once()


def test_process_pages(config: Config, pdf_file: str):
    stage = ArxivSource(config, query="unittest", cache_dir="/does/not/exist")
    documents = stage._process_pages(pdf_file)

    assert len(documents) == 1
    document = documents[0]
    assert document.page_content == 'Morpheus\nunittest'
    assert document.metadata == {'source': pdf_file, 'page': 0}


def test_process_pages_error(config: Config, tmp_path: str):
    bad_pdf_filename = os.path.join(tmp_path, "bad.pdf")
    with open(bad_pdf_filename, 'w', encoding='utf-8') as fh:
        fh.write("Not a PDF")

    stage = ArxivSource(config, query="unittest", cache_dir=tmp_path)

    with pytest.raises(RuntimeError):
        stage._process_pages(bad_pdf_filename)


@mock.patch("langchain_community.document_loaders.PyPDFLoader")
def test_process_pages_retry(mock_pdf_loader: mock.MagicMock, config: Config, pypdf: types.ModuleType):
    call_count = 0

    def mock_load():
        nonlocal call_count
        call_count += 1
        if call_count < 5:
            raise pypdf.errors.PdfStreamError()

        if call_count == 5:
            return ["unittest"]

        assert False, "Should not be called more than 5 times"

    mock_pdf_loader.return_value = mock_pdf_loader
    mock_pdf_loader.load.side_effect = mock_load
    stage = ArxivSource(config, query="unittest", cache_dir="/does/not/exist")
    documents = stage._process_pages('/does/not/exist/fake.pdf')

    assert call_count == 5

    assert documents == ["unittest"]


@pytest.mark.parametrize("chunk_size", [200, 1000])
def test_splitting_pages(config: Config,
                         pdf_file: str,
                         langchain_community: types.ModuleType,
                         chunk_size: int,
                         dataset_cudf: DatasetManager):
    chunk_overlap = 100
    content = "Morpheus\nunittest"
    while len(content) < chunk_size:
        content += f'\n{string.ascii_lowercase}'

    # The test PDF is quite small, so we need to pad it out to get multiple pages
    # I checked this won't evently divide into either of the two values for chunk_size
    if len(content) != chunk_size:
        split_point = content.rfind('\n')
        first_row = content[0:split_point]
        second_row = content[split_point + 1:]
        while len(second_row) < chunk_overlap:
            split_point = content[0:split_point].rfind('\n')
            second_row = content[split_point + 1:]

        page_content_col = [first_row, second_row]
    else:
        page_content_col = [content]

    num_expected_chunks = len(page_content_col)
    source_col = []
    page_col = []
    type_col = []
    for _ in range(num_expected_chunks):
        source_col.append(pdf_file)
        page_col.append(0)
        type_col.append("Document")

    expected_df = cudf.DataFrame({
        "page_content": page_content_col, "source": source_col, "page": page_col, "type": type_col
    })

    loader = langchain_community.document_loaders.PyPDFLoader(pdf_file)
    documents = loader.load()
    assert len(documents) == 1

    document = documents[0]
    document.page_content = content

    stage = ArxivSource(config, query="unittest", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    msg = stage._splitting_pages(documents)

    dataset_cudf.assert_compare_df(msg.df, expected_df)


def test_splitting_pages_no_chunks(config: Config,
                                   pdf_file: str,
                                   langchain_community: types.ModuleType,
                                   dataset_cudf: DatasetManager):
    content = "Morpheus\nunittest"
    page_content_col = [content]
    source_col = [pdf_file]
    page_col = [0]
    type_col = ["Document"]

    expected_df = cudf.DataFrame({
        "page_content": page_content_col, "source": source_col, "page": page_col, "type": type_col
    })

    loader = langchain_community.document_loaders.PyPDFLoader(pdf_file)
    documents = loader.load()
    assert len(documents) == 1

    document = documents[0]
    document.page_content = content

    stage = ArxivSource(config, query="unittest")
    msg = stage._splitting_pages(documents)

    dataset_cudf.assert_compare_df(msg.df, expected_df)
