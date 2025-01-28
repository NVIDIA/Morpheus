# Copyright (c) 2022-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import typing
from unittest import mock

from morpheus.messages import ControlMessage
from morpheus_llm.llm import InputMap
from morpheus_llm.llm import LLMContext
from morpheus_llm.llm import LLMNodeBase
from morpheus_llm.llm import LLMTask
from morpheus_llm.llm import LLMTaskHandler


def _mk_context(parent_context: LLMContext, input_values: dict) -> LLMContext:
    inputs: list[InputMap] = []

    for input_name, input_value in input_values.items():
        inputs.append(InputMap(f"/{input_name}", input_name))
        parent_context.set_output(input_name, input_value)

    return parent_context.push("test", inputs)


def execute_node(node: LLMNodeBase,
                 task_dict: dict = None,
                 input_message: ControlMessage = None,
                 **input_values: dict) -> typing.Any:
    """
    Executes an LLM Node with the necessary LLM context, and extracts the output values.
    """
    task_dict = task_dict or {}
    input_message = input_message or ControlMessage()
    task = LLMTask("unittests", task_dict)
    parent_context = LLMContext(task, input_message)

    context = _mk_context(parent_context, input_values)

    async def execute():
        # `asyncio.run(obj)`` will raise a `ValueError`` if `asyncio.iscoutine(obj)` is `False` for composite nodes
        # that don't directly implement `execute()` this causes a failure because while
        # `mrc.core.coro.CppToPyAwaitable` is awaitable it is not a coroutine.

        return await node.execute(context)

    context = asyncio.run(execute())

    return context.view_outputs


def execute_task_handler(task_handler: LLMTaskHandler,
                         task_dict: dict,
                         input_message: ControlMessage,
                         **input_values: dict) -> ControlMessage:
    """
    Executes an LLM task handler with the necessary LLM context.
    """
    task = LLMTask("unittests", task_dict)
    parent_context = LLMContext(task, input_message)
    context = _mk_context(parent_context, input_values)

    message = asyncio.run(task_handler.try_handle(context))

    return message


def _mk_mock_choice(message: str) -> mock.MagicMock:
    mock_choice = mock.MagicMock()
    mock_choice.message.content = message
    return mock_choice


def mk_mock_openai_response(messages: list[str]) -> mock.MagicMock:
    """
    Creates a mocked openai.types.chat.chat_completion.ChatCompletion response with the given messages.
    """
    response = mock.MagicMock()

    response.choices = [_mk_mock_choice(message) for message in messages]

    response_dict = {"choices": [{'message': {'role': 'assistant', 'content': message}} for message in messages]}

    response.dict.return_value = response_dict
    response.model_dump.return_value = response_dict

    return response


def mk_mock_langchain_tool(responses: list[str]) -> mock.MagicMock:
    """
    Creates a mocked LangChainTestTool with the given responses.
    """

    # Langchain will call inspect.signature on the tool methods, typically mock objects don't have a signature,
    # explicitly providing one here
    async def _arun_spec(*_, **__):
        pass

    def run_spec(*_, **__):
        pass

    tool = mock.MagicMock()
    tool.arun = mock.create_autospec(spec=_arun_spec)
    tool.arun.side_effect = responses
    tool.run = mock.create_autospec(run_spec)
    tool.run.side_effect = responses
    return tool
