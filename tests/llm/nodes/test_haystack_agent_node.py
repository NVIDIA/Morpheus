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
from morpheus.llm.nodes.haystack_agent_node import HaystackAgentNode


def test_constructor(mock_haystack_agent: mock.MagicMock):
    node = HaystackAgentNode(agent=mock_haystack_agent)
    assert isinstance(node, LLMNodeBase)


def test_get_input_names(mock_haystack_agent: mock.MagicMock):
    node = HaystackAgentNode(agent=mock_haystack_agent)
    assert node.get_input_names() == ["query"]


@pytest.mark.parametrize("values, return_only_answer, expected_calls",
                         [({
                             'query': "test query"
                         }, False, [mock.call(query="test query")]),
                          ({
                              'query': ["test query"]
                          }, True, [mock.call({"query": "test query"})])])
def test_execute(
    mock_haystack_agent: mock.MagicMock,
    values: dict,
    return_only_answer: bool,
    mock_haystack_agent_run_return: dict,
    mock_haystack_answer_data: dict,
    expected_calls: list[mock.call],
):
    if not return_only_answer:
        expected_output = {
            'query': 'test query', 'answers': [mock_haystack_answer_data], 'transcript': 'query transcript'
        }
    else:
        expected_output = {'answers': [mock_haystack_answer_data['answer']]}

    mock_haystack_agent.run.return_value = mock_haystack_agent_run_return

    node = HaystackAgentNode(agent=mock_haystack_agent, return_only_answer=return_only_answer)
    assert execute_node(node, **values) == [expected_output]
    mock_haystack_agent.run.assert_has_calls(expected_calls)
