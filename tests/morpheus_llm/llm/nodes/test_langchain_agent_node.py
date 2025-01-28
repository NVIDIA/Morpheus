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

import re
import typing
from operator import itemgetter
from unittest import mock

import pytest

from _utils.llm import execute_node
from _utils.llm import mk_mock_langchain_tool
from _utils.llm import mk_mock_openai_response
from morpheus_llm.llm import LLMNodeBase
from morpheus_llm.llm.nodes.langchain_agent_node import LangChainAgentNode

try:
    from langchain.agents import AgentType
    from langchain.agents import Tool
    from langchain.agents import initialize_agent
    from langchain.callbacks.manager import AsyncCallbackManagerForToolRun
    from langchain.callbacks.manager import CallbackManagerForToolRun
    from langchain_core.tools import BaseTool
    from langchain_openai import ChatOpenAI
except ImportError:
    pass


class OutputParserExceptionStandin(Exception):
    """
    Stand-in for the OutputParserException class to avoid importing the actual class from the langchain_core.exceptions.
    There is a need to have OutputParserException objects appear in test parameters, but we don't want to import
    langchain_core at the top of the test as it is an optional dependency.
    """
    pass


def test_constructor(mock_agent_executor: mock.MagicMock):
    node = LangChainAgentNode(agent_executor=mock_agent_executor)
    assert isinstance(node, LLMNodeBase)


def test_get_input_names(mock_agent_executor: mock.MagicMock):
    node = LangChainAgentNode(agent_executor=mock_agent_executor)
    assert node.get_input_names() == ["prompt"]


@pytest.mark.parametrize(
    "values,arun_return,expected_output,expected_calls",
    [({
        'prompt': "prompt1"
    }, list(range(3)), list(range(3)), [mock.call(prompt="prompt1", metadata=None)]),
     ({
         'a': ['b', 'c', 'd'], 'c': ['d', 'e', 'f'], 'e': ['f', 'g', 'h']
     },
      list(range(3)), [list(range(3))] * 3,
      [
          mock.call(a='b', c='d', e='f', metadata=None),
          mock.call(a='c', c='e', e='g', metadata=None),
          mock.call(a='d', c='f', e='h', metadata=None)
      ])],
    ids=["not-lists", "all-lists"])
def test_execute(
    mock_agent_executor: mock.MagicMock,
    values: dict,
    arun_return: list,
    expected_output: list,
    expected_calls: list[mock.call],
):
    # Tests the execute method of the LangChainAgentNode with a mocked agent_executor
    mock_agent_executor.arun.return_value = arun_return

    node = LangChainAgentNode(agent_executor=mock_agent_executor)
    assert execute_node(node, **values) == expected_output
    mock_agent_executor.arun.assert_has_calls(expected_calls)


def test_execute_tools(mock_chat_completion: tuple[mock.MagicMock, mock.MagicMock]):
    # Tests the execute method of the LangChainAgentNode with a a mocked tools and chat completion
    (_, mock_async_client) = mock_chat_completion
    chat_responses = [
        'I should check Tool1\nAction: Tool1\nAction Input: "name a reptile"',
        'I should check Tool2\nAction: Tool2\nAction Input: "name of a day of the week"',
        'I should check Tool1\nAction: Tool1\nAction Input: "name a reptile"',
        'I should check Tool2\nAction: Tool2\nAction Input: "name of a day of the week"',
        'Observation: Answer: Yes!\nI now know the final answer.\nFinal Answer: Yes!'
    ]
    mock_responses = [mk_mock_openai_response([response]) for response in chat_responses]
    mock_async_client.chat.completions.create.side_effect = mock_responses

    llm_chat = ChatOpenAI(model="fake-model", openai_api_key="fake-key")

    mock_tool1 = mk_mock_langchain_tool(["lizard", "frog"])
    mock_tool2 = mk_mock_langchain_tool(["Tuesday", "Thursday"])

    tools = [
        Tool(name="Tool1",
             func=mock_tool1.run,
             coroutine=mock_tool1.arun,
             description="useful for when you need to know the name of a reptile"),
        Tool(name="Tool2",
             func=mock_tool2.run,
             coroutine=mock_tool2.arun,
             description="useful for when you need to know the day of the week")
    ]

    agent = initialize_agent(tools,
                             llm_chat,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True,
                             handle_parsing_errors=True,
                             early_stopping_method="generate",
                             return_intermediate_steps=False)

    node = LangChainAgentNode(agent_executor=agent)

    assert execute_node(node, input="input1") == "Yes!"


def test_execute_error(mock_chat_completion: tuple[mock.MagicMock, mock.MagicMock]):
    # Tests the execute method of the LangChainAgentNode with a a mocked tools and chat completion
    (_, mock_async_client) = mock_chat_completion
    chat_responses = [
        'I should check Tool1\nAction: Tool1\nAction Input: "name a reptile"',
        'I should check Tool2\nAction: Tool2\nAction Input: "name of a day of the week"',
        'Observation: Answer: Yes!\nI now know the final answer.\nFinal Answer: Yes!'
    ]
    mock_responses = [mk_mock_openai_response([response]) for response in chat_responses]
    mock_async_client.chat.completions.create.side_effect = mock_responses

    llm_chat = ChatOpenAI(model="fake-model", openai_api_key="fake-key")

    mock_tool1 = mk_mock_langchain_tool(["lizard"])
    mock_tool2 = mk_mock_langchain_tool(RuntimeError("unittest"))

    tools = [
        Tool(name="Tool1",
             func=mock_tool1.run,
             coroutine=mock_tool1.arun,
             description="useful for when you need to know the name of a reptile"),
        Tool(name="Tool2",
             func=mock_tool2.run,
             coroutine=mock_tool2.arun,
             description="useful for when you need to test tool errors")
    ]

    agent = initialize_agent(tools,
                             llm_chat,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True,
                             handle_parsing_errors=True,
                             early_stopping_method="generate",
                             return_intermediate_steps=False)

    node = LangChainAgentNode(agent_executor=agent)
    assert isinstance(execute_node(node, input="input1"), RuntimeError)


@pytest.mark.parametrize("metadata",
                         [{
                             "morpheus": "unittest"
                         }, {
                             "morpheus": ["unittest"]
                         }, {
                             "morpheus": [f"unittest_{i}" for i in range(3)]
                         }],
                         ids=["single-metadata", "single-metadata-list", "multiple-metadata-list"])
def test_metadata(mock_chat_completion: tuple[mock.MagicMock, mock.MagicMock], metadata: dict):

    class MetadataSaverTool(BaseTool):
        # The base class defines *args and **kwargs in the signature for _run and _arun requiring the arguments-differ
        # pylint: disable=arguments-differ
        name: str = "MetadataSaverTool"
        description: str = "useful for when you need to know the name of a reptile"

        saved_metadata: list[dict] = []

        def _run(
            self,
            query: str,
            run_manager: typing.Optional[CallbackManagerForToolRun] = None,
        ) -> str:
            raise NotImplementedError("This tool only supports async")

        async def _arun(
            self,
            query: str,
            run_manager: typing.Optional[AsyncCallbackManagerForToolRun] = None,
        ) -> str:
            assert query is not None  # avoiding unused-argument
            assert run_manager is not None
            self.saved_metadata.append(run_manager.metadata.copy())
            return "frog"

    if isinstance(metadata['morpheus'], list):
        num_meta = len(metadata['morpheus'])
        input_data = [f"input_{i}" for i in range(num_meta)]
        expected_result = [f"{input_val}: Yes!" for input_val in input_data]
        expected_saved_metadata = [{"morpheus": meta} for meta in metadata['morpheus']]
        response_per_input_counter = {input_val: 0 for input_val in input_data}
    else:
        num_meta = 1
        input_data = "input_0"
        expected_result = "input_0: Yes!"
        expected_saved_metadata = [metadata.copy()]
        response_per_input_counter = {input_data: 0}

    check_tool_response = 'I should check Tool1\nAction: MetadataSaverTool\nAction Input: "name a reptile"'
    final_response = 'Observation: Answer: Yes!\nI now know the final answer.\nFinal Answer: {}: Yes!'

    # Tests the execute method of the LangChainAgentNode with a a mocked tools and chat completion
    (_, mock_async_client) = mock_chat_completion

    # Regex to find the actual prompt from the input which includes the REACT and tool description boilerplate
    input_re = re.compile(r'^Question: (input_\d+)$', re.MULTILINE)

    def mock_llm_chat(*_, messages, **__):
        """
        This method avoids a race condition when running in aysnc mode over multiple inputs. Ensuring that the final
        response is only given for an input after the initial check tool response.
        """

        query = None
        for msg in messages:
            if msg['role'] == 'user':
                query = msg['content']

        assert query is not None

        match = input_re.search(query)
        assert match is not None

        input_key = match.group(1)

        call_count = response_per_input_counter[input_key]

        if call_count == 0:
            response = check_tool_response
        else:
            response = final_response.format(input_key)

        response_per_input_counter[input_key] += 1

        return mk_mock_openai_response([response])

    mock_async_client.chat.completions.create.side_effect = mock_llm_chat

    llm_chat = ChatOpenAI(model="fake-model", openai_api_key="fake-key")

    metadata_saver_tool = MetadataSaverTool()

    tools = [metadata_saver_tool]

    agent = initialize_agent(tools,
                             llm_chat,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True,
                             handle_parsing_errors=True,
                             early_stopping_method="generate",
                             return_intermediate_steps=False)

    node = LangChainAgentNode(agent_executor=agent)

    assert execute_node(node, input=input_data, metadata=metadata) == expected_result

    # Since we are running in async mode, we will need to sort saved metadata
    assert sorted(metadata_saver_tool.saved_metadata, key=itemgetter('morpheus')) == expected_saved_metadata


@pytest.mark.parametrize(
    "arun_return,replace_value,expected_output",
    [
        (
            [[OutputParserExceptionStandin("Parsing Error"), "A valid result."]],
            "Default error message.",
            [["Default error message.", "A valid result."]],
        ),
        (
            [["A valid result."], [Exception("General error"), "Another valid result."]],
            "Another default error message.",
            [["A valid result."], ["Another default error message.", "Another valid result."]],
        ),
        (
            [
                ["A valid result.", OutputParserExceptionStandin("Parsing Error")],
                [Exception("General error"), "Another valid result."],
            ],
            None,
            [["A valid result.", None], [None, "Another valid result."]],
        ),
    ],
    ids=["parsing_error_handling", "exception_handling", "none_as_replacement_value"],
)
def test_execute_replaces_exceptions(
    mock_agent_executor: mock.MagicMock,
    arun_return: list,
    replace_value: str,
    expected_output: list,
):
    # We couldn't import OutputParserException at the module level, so we need to replace instances of
    # OutputParserExceptionStandin with OutputParserException
    from langchain_core.exceptions import OutputParserException

    arun_return_tmp = []
    for values in arun_return:
        values_tmp = []
        for value in values:
            if isinstance(value, OutputParserExceptionStandin):
                values_tmp.append(OutputParserException(*value.args))
            else:
                values_tmp.append(value)
        arun_return_tmp.append(values_tmp)

    arun_return = arun_return_tmp

    placeholder_input_values = {"foo": "bar"}  # a non-empty placeholder input for the context
    mock_agent_executor.arun.return_value = arun_return

    node = LangChainAgentNode(
        agent_executor=mock_agent_executor,
        replace_exceptions=True,
        replace_exceptions_value=replace_value,
    )
    output = execute_node(node, **placeholder_input_values)
    assert output == expected_output
