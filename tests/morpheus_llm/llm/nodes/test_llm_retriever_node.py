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

import typing
from unittest import mock

import pytest

from _utils.llm import execute_node
from morpheus_llm.llm import LLMNodeBase
from morpheus_llm.llm.nodes.retriever_node import RetrieverNode


@pytest.mark.parametrize("embedding", [None, mock.AsyncMock()])
def test_constructor(embedding: typing.Callable | None):
    mock_vdb_service = mock.MagicMock()
    node = RetrieverNode(embedding=embedding, service=mock_vdb_service)
    assert isinstance(node, LLMNodeBase)


@pytest.mark.parametrize("embedding, expected", [(None, ["embedding"]), (mock.AsyncMock(), ["query"])])
def test_get_input_names(embedding: typing.Callable | None, expected: list[str]):
    mock_vdb_service = mock.MagicMock()
    node = RetrieverNode(embedding=embedding, service=mock_vdb_service)
    assert node.get_input_names() == expected


@pytest.mark.parametrize("embedding", [None, mock.AsyncMock(return_value=[[1.2, 2.3, 3.4], [4.5, 5.6, 6.7]])])
def test_execute(embedding: mock.AsyncMock | None):
    mock_vdb_service = mock.MagicMock()
    mock_vdb_service.similarity_search = mock.AsyncMock(return_value=[[1, 2, 3], [4, 5, 6]])

    node = RetrieverNode(embedding=embedding, service=mock_vdb_service)

    expected_output = [[1, 2, 3], [4, 5, 6]]

    assert execute_node(node, query=["query"]) == expected_output
