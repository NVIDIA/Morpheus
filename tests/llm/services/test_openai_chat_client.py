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

import asyncio
from unittest import mock

import pytest

from morpheus.llm.services.llm_service import LLMClient
from morpheus.llm.services.openai_chat_service import OpenAIChatClient
from morpheus.llm.services.openai_chat_service import OpenAIChatService


@pytest.fixture(name="openai_chat_service")
def openai_chat_service_fixture():
    return OpenAIChatService()


def test_constructor(mock_openai: mock.MagicMock, openai_chat_service: OpenAIChatService):
    client = OpenAIChatClient(openai_chat_service, model_name="test_model")
    assert isinstance(client, LLMClient)
    mock_openai.chart.completions.create.assert_not_called()


@pytest.mark.parametrize("use_async", [True, False])
@pytest.mark.parametrize(
    "input_dict, set_assistant, expected_messages",
    [({
        "prompt": "test_prompt", "assistant": "assistant_response"
    },
      True,
      [{
          "role": "system", "content": "You are a helpful assistant."
      }, {
          "role": "user", "content": "test_prompt"
      }, {
          "role": "assistant", "content": "assistant_response"
      }]),
     ({
         "prompt": "test_prompt"
     },
      False, [{
          "role": "system", "content": "You are a helpful assistant."
      }, {
          "role": "user", "content": "test_prompt"
      }])])
@pytest.mark.parametrize("temperature", [0, 1, 2])
def test_generate(mock_openai: mock.MagicMock,
                  mock_async_openai: mock.MagicMock,
                  openai_chat_service: OpenAIChatService,
                  use_async: bool,
                  input_dict: dict[str, str],
                  set_assistant: bool,
                  expected_messages: list[dict],
                  temperature: int):

    client = OpenAIChatClient(openai_chat_service,
                              model_name="test_model",
                              set_assistant=set_assistant,
                              temperature=temperature)

    if use_async:
        results = asyncio.run(client.generate_async(input_dict))
        mock_async_openai.chat.completions.create.assert_called_once_with(model="test_model",
                                                                          messages=expected_messages,
                                                                          temperature=temperature)

    else:
        results = client.generate(input_dict)
        mock_openai.chat.completions.create.assert_called_once_with(model="test_model",
                                                                    messages=expected_messages,
                                                                    temperature=temperature)

    assert results == "test_output"


@pytest.mark.parametrize("use_async", [True, False])
@pytest.mark.parametrize(
    "inputs, set_assistant, expected_messages",
    [({
        "prompt": ["prompt1", "prompt2"], "assistant": ["assistant1", "assistant2"]
    },
      True,
      [[{
          "role": "system", "content": "You are a helpful assistant."
      }, {
          "role": "user", "content": "prompt1"
      }, {
          "role": "assistant", "content": "assistant1"
      }],
       [{
           "role": "system", "content": "You are a helpful assistant."
       }, {
           "role": "user", "content": "prompt2"
       }, {
           "role": "assistant", "content": "assistant2"
       }]]),
     ({
         "prompt": ["prompt1", "prompt2"]
     },
      False,
      [[{
          "role": "system", "content": "You are a helpful assistant."
      }, {
          "role": "user", "content": "prompt1"
      }], [{
          "role": "system", "content": "You are a helpful assistant."
      }, {
          "role": "user", "content": "prompt2"
      }]])])
@pytest.mark.parametrize("temperature", [0, 1, 2])
def test_generate_batch(mock_openai: mock.MagicMock,
                        mock_async_openai: mock.MagicMock,
                        openai_chat_service: OpenAIChatService,
                        use_async: bool,
                        inputs: dict[str, list[str]],
                        set_assistant: bool,
                        expected_messages: list[list[dict]],
                        temperature: int):
    client = OpenAIChatClient(openai_chat_service,
                              model_name="test_model",
                              set_assistant=set_assistant,
                              temperature=temperature)

    expected_results = ["test_output" for _ in range(len(inputs["prompt"]))]

    if use_async:
        results = asyncio.run(client.generate_batch_async(inputs))
        mock_openai_instance = mock_async_openai
    else:
        results = client.generate_batch(inputs)
        mock_openai_instance = mock_openai

    for messages in expected_messages:
        mock_openai_instance.chat.completions.create.assert_any_call(model="test_model",
                                                                     messages=messages,
                                                                     temperature=temperature)

    assert mock_openai_instance.chat.completions.create.call_count == len(expected_messages)
    assert results == expected_results
