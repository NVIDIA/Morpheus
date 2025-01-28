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
from morpheus_llm.llm.nodes.rag_node import RAGNode


@pytest.mark.parametrize("embedding", [None, mock.AsyncMock()])
@pytest.mark.parametrize("prompt,template_format",
                         [("contexts={contexts} query={query}", "f-string"),
                          ("contexts={{ contexts }} query={{ query }}", "jinja")])
def test_constructor(prompt: str,
                     template_format: str,
                     embedding: typing.Callable | None,
                     mock_llm_client: mock.MagicMock):
    node = RAGNode(prompt=prompt,
                   template_format=template_format,
                   vdb_service=mock.MagicMock(),
                   embedding=embedding,
                   llm_client=mock_llm_client)
    assert isinstance(node, LLMNodeBase)


@pytest.mark.parametrize("embedding, expected_inputs", [(None, ['embedding', 'query']), (mock.AsyncMock(), ["query"])])
def test_get_input_names(embedding: typing.Callable | None, expected_inputs: list[str],
                         mock_llm_client: mock.MagicMock):
    node = RAGNode(prompt="contexts={contexts} query={query}",
                   template_format="f-string",
                   vdb_service=mock.MagicMock(),
                   embedding=embedding,
                   llm_client=mock_llm_client)
    assert sorted(node.get_input_names()) == sorted(expected_inputs)


def test_execute(mock_llm_client: mock.MagicMock):
    mock_embedding = mock.AsyncMock(return_value=[[1.2, 2.3, 3.4], [4.5, 5.6, 6.7]])

    mock_vdb_service = mock.MagicMock()
    mock_vdb_service.similarity_search = mock.AsyncMock(return_value=[[1, 2, 3], [4, 5, 6]])

    mock_llm_client.generate_batch_async.return_value = ["response1", "response2"]

    node = RAGNode(prompt="contexts={contexts} query={query}",
                   template_format="f-string",
                   vdb_service=mock_vdb_service,
                   embedding=mock_embedding,
                   llm_client=mock_llm_client)

    expected_output = {
        'generate': ["response1", "response2"],
        'prompt': ['contexts=[1, 2, 3] query=query1', 'contexts=[4, 5, 6] query=query2'],
        'retriever': [[1, 2, 3], [4, 5, 6]]
    }

    assert execute_node(node, query=["query1", "query2"]) == expected_output
