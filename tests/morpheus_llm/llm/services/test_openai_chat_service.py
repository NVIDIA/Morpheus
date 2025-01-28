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

import asyncio
import os
import time
from unittest import mock

import pytest

from _utils.llm import mk_mock_openai_response
from morpheus_llm.llm.services.openai_chat_service import OpenAIChatService


@pytest.fixture(name="set_default_openai_api_key", autouse=True, scope="function")
def set_default_openai_api_key_fixture():
    # Must have an API key set to create the openai client
    with mock.patch.dict(os.environ, clear=True, values={"OPENAI_API_KEY": "testing_api_key"}):
        yield


def assert_called_once_with_relaxed(mock_obj, *args, **kwargs):

    if (len(mock_obj.call_args_list) == 1):

        recent_call = mock_obj.call_args_list[-1]

        # Ensure that the number of arguments matches by adding ANY to the back of the args
        if (len(args) < len(recent_call.args)):
            args = tuple(list(args) + [mock.ANY] * (len(recent_call.args) - len(args)))

        addl_kwargs = {key: mock.ANY for key in recent_call.kwargs.keys() if key not in kwargs}

        kwargs.update(addl_kwargs)

    mock_obj.assert_called_once_with(*args, **kwargs)


@pytest.mark.parametrize("api_key", ["12345", None])
@pytest.mark.parametrize("base_url", ["http://test.openai.com/v1", None])
@pytest.mark.parametrize("org_id", ["my-org-124", None])
def test_constructor(mock_chat_completion: tuple[mock.MagicMock, mock.MagicMock],
                     api_key: str,
                     base_url: str,
                     org_id: str):

    OpenAIChatService(api_key=api_key, base_url=base_url, org_id=org_id).get_client(model_name="test_model")

    if (api_key is None):
        api_key = os.environ["OPENAI_API_KEY"]

    for mock_client in mock_chat_completion:
        assert_called_once_with_relaxed(mock_client, organization=org_id, api_key=api_key, base_url=base_url)


@pytest.mark.parametrize("max_retries", [5, 10, -1, None])
def test_max_retries(mock_chat_completion: tuple[mock.MagicMock, mock.MagicMock], max_retries: int):
    OpenAIChatService().get_client(model_name="test_model", max_retries=max_retries)

    for mock_client in mock_chat_completion:
        assert_called_once_with_relaxed(mock_client, max_retries=max_retries)


@pytest.mark.parametrize("use_json", [True, False])
def test_client_json(mock_chat_completion: tuple[mock.MagicMock, mock.MagicMock], use_json: bool):
    client = OpenAIChatService().get_client(model_name="test_model", json=use_json)

    # Perform a dummy generate call
    client.generate(prompt="test_prompt")

    if (use_json):
        assert_called_once_with_relaxed(mock_chat_completion[0].chat.completions.create,
                                        response_format={"type": "json_object"})
    else:
        assert mock_chat_completion[0].chat.completions.create.call_args_list[-1].kwargs.get("response_format") is None


@pytest.mark.parametrize("set_assistant", [True, False])
def test_client_set_assistant(mock_chat_completion: tuple[mock.MagicMock, mock.MagicMock], set_assistant: bool):
    client = OpenAIChatService().get_client(model_name="test_model", set_assistant=set_assistant)

    # Perform a dummy generate call
    client.generate(prompt="test_prompt", assistant="assistant_message")

    messages = mock_chat_completion[0].chat.completions.create.call_args_list[-1].kwargs["messages"]

    found_assistant = False

    for message in messages:
        if (message.get("role") == "assistant"):
            found_assistant = True
            break

    assert found_assistant == set_assistant


@pytest.mark.parametrize("use_async", [True, False])
@pytest.mark.parametrize(
    "input_dict, set_assistant, expected_messages",
    [({
        "prompt": "test_prompt", "assistant": "assistant_response"
    },
      True, [{
          "role": "user", "content": "test_prompt"
      }, {
          "role": "assistant", "content": "assistant_response"
      }]), ({
          "prompt": "test_prompt"
      }, False, [{
          "role": "user", "content": "test_prompt"
      }])])
@pytest.mark.parametrize("temperature", [0, 1, 2])
def test_generate(mock_chat_completion: tuple[mock.MagicMock, mock.MagicMock],
                  use_async: bool,
                  input_dict: dict[str, str],
                  set_assistant: bool,
                  expected_messages: list[dict],
                  temperature: int):
    (mock_client, mock_async_client) = mock_chat_completion
    client = OpenAIChatService().get_client(model_name="test_model",
                                            set_assistant=set_assistant,
                                            temperature=temperature)

    if use_async:
        results = asyncio.run(client.generate_async(**input_dict))
        mock_async_client.chat.completions.create.assert_called_once_with(model="test_model",
                                                                          messages=expected_messages,
                                                                          temperature=temperature)
        mock_client.chat.completions.create.assert_not_called()

    else:
        results = client.generate(**input_dict)
        mock_client.chat.completions.create.assert_called_once_with(model="test_model",
                                                                    messages=expected_messages,
                                                                    temperature=temperature)
        mock_async_client.chat.completions.create.assert_not_called()

    assert results == "test_output"


@pytest.mark.parametrize("use_async", [True, False])
@pytest.mark.parametrize("inputs, set_assistant, expected_messages",
                         [({
                             "prompt": ["prompt1", "prompt2"], "assistant": ["assistant1", "assistant2"]
                         },
                           True,
                           [[{
                               "role": "user", "content": "prompt1"
                           }, {
                               "role": "assistant", "content": "assistant1"
                           }], [{
                               "role": "user", "content": "prompt2"
                           }, {
                               "role": "assistant", "content": "assistant2"
                           }]]),
                          ({
                              "prompt": ["prompt1", "prompt2"]
                          },
                           False, [[{
                               "role": "user", "content": "prompt1"
                           }], [{
                               "role": "user", "content": "prompt2"
                           }]])])
@pytest.mark.parametrize("temperature", [0, 1, 2])
def test_generate_batch(mock_chat_completion: tuple[mock.MagicMock, mock.MagicMock],
                        use_async: bool,
                        inputs: dict[str, list[str]],
                        set_assistant: bool,
                        expected_messages: list[list[dict]],
                        temperature: int):
    (mock_client, mock_async_client) = mock_chat_completion
    client = OpenAIChatService().get_client(model_name="test_model",
                                            set_assistant=set_assistant,
                                            temperature=temperature)

    expected_results = ["test_output" for _ in range(len(inputs["prompt"]))]
    expected_calls = [
        mock.call(model="test_model", messages=messages, temperature=temperature) for messages in expected_messages
    ]

    if use_async:
        results = asyncio.run(client.generate_batch_async(inputs))
        mock_async_client.chat.completions.create.assert_has_calls(expected_calls, any_order=False)
        mock_client.chat.completions.create.assert_not_called()

    else:
        results = client.generate_batch(inputs)
        mock_client.chat.completions.create.assert_has_calls(expected_calls, any_order=False)
        mock_async_client.chat.completions.create.assert_not_called()

    assert results == expected_results


def test_generate_batch_exception(mock_chat_completion: tuple[mock.MagicMock, mock.MagicMock], ):

    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    (mock_client, _) = mock_chat_completion

    def mock_create(*_, messages, **__):

        choices = []

        for i, p in enumerate(messages):
            if ("error" in p["content"]):
                raise RuntimeError("unittest")

            choices.append(
                Choice(index=i,
                       finish_reason="stop",
                       message=ChatCompletionMessage(content=p["content"], role="assistant")))

        return ChatCompletion(id="test_id",
                              model="test_model",
                              object="chat.completion",
                              choices=choices,
                              created=int(time.time()))

    mock_client.chat.completions.create.side_effect = mock_create

    client = OpenAIChatService().get_client(model_name="test_model")

    # Return_exceptions=True
    results = list(client.generate_batch({'prompt': ["prompt1", "error", "prompt3"]}, return_exceptions=True))
    assert isinstance(results[1], RuntimeError)

    # Test exceptions, return_exceptions=False
    with pytest.raises(RuntimeError, match="unittest"):
        client.generate_batch({'prompt': ["prompt1", "error", "prompt3"]}, return_exceptions=False)


async def test_generate_batch_async_exception(mock_chat_completion: tuple[mock.MagicMock, mock.MagicMock], ):

    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    (_, mock_async_client) = mock_chat_completion

    async def mock_create(*_, messages, **__):

        choices = []

        for i, p in enumerate(messages):
            if ("error" in p["content"]):
                raise RuntimeError("unittest")

            choices.append(
                Choice(index=i,
                       finish_reason="stop",
                       message=ChatCompletionMessage(content=p["content"], role="assistant")))

        return ChatCompletion(id="test_id",
                              model="test_model",
                              object="chat.completion",
                              choices=choices,
                              created=int(time.time()))

    mock_async_client.chat.completions.create.side_effect = mock_create

    client = OpenAIChatService().get_client(model_name="test_model")

    # Return_exceptions=True
    results = await client.generate_batch_async({'prompt': ["prompt1", "error", "prompt3"]}, return_exceptions=True)
    assert isinstance(results[1], RuntimeError)

    # Test exceptions, return_exceptions=False
    with pytest.raises(RuntimeError, match="unittest"):
        await client.generate_batch_async({'prompt': ["prompt1", "error", "prompt3"]}, return_exceptions=False)


@pytest.mark.parametrize("completion", [[], [None]], ids=["no_choices", "no_content"])
@pytest.mark.usefixtures("mock_chat_completion")
def test_extract_completion_errors(completion: list):
    client = OpenAIChatService().get_client(model_name="test_model")
    mock_completion = mk_mock_openai_response(completion)

    with pytest.raises(ValueError):
        client._extract_completion(mock_completion)


def test_get_client():
    service = OpenAIChatService()
    client = service.get_client(model_name="test_model")

    assert client.model_name == "test_model"

    client = service.get_client(model_name="test_model2", extra_arg="test_arg")

    assert client.model_name == "test_model2"
    assert client.model_kwargs == {"extra_arg": "test_arg"}


@pytest.mark.parametrize("temperature", [0, 1, 2])
@pytest.mark.parametrize("max_retries", [5, 10])
def test_get_client_passed_args(mock_chat_completion: tuple[mock.MagicMock, mock.MagicMock],
                                temperature: int,
                                max_retries: int):
    service = OpenAIChatService()
    client = service.get_client(model_name="test_model", temperature=temperature, test='this', max_retries=max_retries)

    # Perform a dummy generate call
    client.generate(prompt="test_prompt")

    # Ensure the get_client method passed on the set_assistant and model kwargs
    assert_called_once_with_relaxed(mock_chat_completion[0].chat.completions.create,
                                    model="test_model",
                                    temperature=temperature,
                                    test='this')
