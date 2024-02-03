# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest import mock

import pytest

from _utils.llm import execute_node
from morpheus.llm import LLMNodeBase
from morpheus.llm.nodes.langchain_agent_node import LangChainAgentNode


def test_constructor(mock_langchain_agent_executor: mock.MagicMock):
    node = LangChainAgentNode(agent_executor=mock_langchain_agent_executor)
    assert isinstance(node, LLMNodeBase)


def test_get_input_names(mock_langchain_agent_executor: mock.MagicMock):
    node = LangChainAgentNode(agent_executor=mock_langchain_agent_executor)
    assert node.get_input_names() == ["prompt"]


@pytest.mark.parametrize(
    "values,arun_return,expected_output,expected_calls",
    [({
        'prompt': "prompt1"
    }, list(range(3)), list(range(3)), [mock.call(prompt="prompt1")]),
     ({
         'a': ['b', 'c', 'd'], 'c': ['d', 'e', 'f'], 'e': ['f', 'g', 'h']
     },
      list(range(3)), [list(range(3))] * 3,
      [mock.call(a='b', c='d', e='f'), mock.call(a='c', c='e', e='g'), mock.call(a='d', c='f', e='h')])],
    ids=["not-lists", "all-lists"])
def test_execute(
    mock_langchain_agent_executor: mock.MagicMock,
    values: dict,
    arun_return: list,
    expected_output: list,
    expected_calls: list[mock.call],
):
    mock_langchain_agent_executor.arun.return_value = arun_return

    node = LangChainAgentNode(agent_executor=mock_langchain_agent_executor)
    assert execute_node(node, **values) == expected_output
    mock_langchain_agent_executor.arun.assert_has_calls(expected_calls)
