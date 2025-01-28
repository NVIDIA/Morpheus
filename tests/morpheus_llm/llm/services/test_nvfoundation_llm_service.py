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
from unittest import mock

import pytest

from morpheus_llm.llm.services.nvfoundation_llm_service import NVFoundationLLMClient
from morpheus_llm.llm.services.nvfoundation_llm_service import NVFoundationLLMService

try:
    from langchain_core.messages import ChatMessage
    from langchain_core.outputs import ChatGeneration
    from langchain_core.outputs import LLMResult
except ImportError:
    pass


@pytest.fixture(name="set_default_nvidia_api_key", autouse=True, scope="function")
def set_default_nvidia_api_key_fixture():
    # Must have an API key set to create the openai client
    with mock.patch.dict(os.environ, clear=True, values={"NVIDIA_API_KEY": "nvapi-testing_api_key"}):
        yield


@pytest.mark.parametrize("api_key", ["nvapi-12345", None])
@pytest.mark.parametrize("base_url", ["http://test.nvidia.com/v1", None])
def test_constructor(api_key: str | None, base_url: bool | None):

    service = NVFoundationLLMService(api_key=api_key, base_url=base_url)

    if (api_key is None):
        api_key = os.environ["NVIDIA_API_KEY"]

    assert service.api_key == api_key
    assert service.base_url == base_url


def test_get_client():
    service = NVFoundationLLMService(api_key="test_api_key")
    client = service.get_client(model_name="test_model")

    assert isinstance(client, NVFoundationLLMClient)


def test_model_kwargs():
    service = NVFoundationLLMService(arg1="default_value1", arg2="default_value2")

    client = service.get_client(model_name="model_name", arg2="value2")

    assert client.model_kwargs["arg1"] == "default_value1"
    assert client.model_kwargs["arg2"] == "value2"


def test_get_input_names():
    client = NVFoundationLLMService().get_client(model_name="test_model", additional_arg="test_arg")

    assert client.get_input_names() == ["prompt"]


def test_generate():
    with mock.patch("langchain_nvidia_ai_endpoints.ChatNVIDIA.generate_prompt", autospec=True) as mock_nvfoundationllm:

        def mock_generation_side_effect(*_, **kwargs):
            return LLMResult(generations=[[
                ChatGeneration(message=ChatMessage(content=x.text, role="assistant")) for x in kwargs["prompts"]
            ]])

        mock_nvfoundationllm.side_effect = mock_generation_side_effect

        client = NVFoundationLLMService().get_client(model_name="test_model")
        assert client.generate(prompt="test_prompt") == "test_prompt"


def test_generate_batch():

    with mock.patch("langchain_nvidia_ai_endpoints.ChatNVIDIA.generate_prompt", autospec=True) as mock_nvfoundationllm:

        def mock_generation_side_effect(*_, **kwargs):
            return LLMResult(generations=[[ChatGeneration(message=ChatMessage(content=x.text, role="assistant"))]
                                          for x in kwargs["prompts"]])

        mock_nvfoundationllm.side_effect = mock_generation_side_effect

        client = NVFoundationLLMService().get_client(model_name="test_model")

        assert client.generate_batch({'prompt': ["prompt1", "prompt2"]}) == ["prompt1", "prompt2"]


def test_generate_batch_exception():

    with mock.patch("langchain_nvidia_ai_endpoints.ChatNVIDIA.generate_prompt", autospec=True) as mock_nvfoundationllm:

        def mock_generation_side_effect(*_, **kwargs):
            generations = []

            for x in kwargs["prompts"]:
                if "error" in x.text:
                    raise RuntimeError("unittest")

                generations.append([ChatGeneration(message=ChatMessage(content=x.text, role="assistant"))])

            return LLMResult(generations=generations)

        mock_nvfoundationllm.side_effect = mock_generation_side_effect

        client = NVFoundationLLMService().get_client(model_name="test_model")

        # Test with return_exceptions=True
        with pytest.warns(UserWarning):
            assert client.generate_batch({'prompt': ["prompt1", "prompt2"]},
                                         return_exceptions=True) == ["prompt1", "prompt2"]

        # Test with return_exceptions=False
        with pytest.raises(RuntimeError, match="unittest"):
            client.generate_batch({'prompt': ["prompt1", "error", "prompt3"]}, return_exceptions=False)


async def test_generate_async():

    with mock.patch("langchain_nvidia_ai_endpoints.ChatNVIDIA.agenerate_prompt", autospec=True) as mock_nvfoundationllm:

        def mock_generation_side_effect(*_, **kwargs):
            return LLMResult(generations=[[ChatGeneration(message=ChatMessage(content=x.text, role="assistant"))]
                                          for x in kwargs["prompts"]])

        mock_nvfoundationllm.side_effect = mock_generation_side_effect

        client = NVFoundationLLMService().get_client(model_name="test_model")

        assert await client.generate_async(prompt="test_prompt") == "test_prompt"


async def test_generate_batch_async():

    with mock.patch("langchain_nvidia_ai_endpoints.ChatNVIDIA.agenerate_prompt", autospec=True) as mock_nvfoundationllm:

        def mock_generation_side_effect(*_, **kwargs):
            return LLMResult(generations=[[ChatGeneration(message=ChatMessage(content=x.text, role="assistant"))]
                                          for x in kwargs["prompts"]])

        mock_nvfoundationllm.side_effect = mock_generation_side_effect

        client = NVFoundationLLMService().get_client(model_name="test_model")

        assert await client.generate_batch_async({'prompt': ["prompt1", "prompt2"]})


async def test_generate_batch_async_exception():

    with mock.patch("langchain_nvidia_ai_endpoints.ChatNVIDIA.agenerate_prompt", autospec=True) as mock_nvfoundationllm:

        async def mock_generation_side_effect(*_, **kwargs):
            generations = []

            for x in kwargs["prompts"]:
                if "error" in x.text:
                    raise RuntimeError("unittest")

                generations.append([ChatGeneration(message=ChatMessage(content=x.text, role="assistant"))])

            return LLMResult(generations=generations)

        mock_nvfoundationllm.side_effect = mock_generation_side_effect

        client = NVFoundationLLMService().get_client(model_name="test_model")

        # Test with return_exceptions=True
        with pytest.warns(UserWarning):
            assert (await client.generate_batch_async({'prompt': ["prompt1", "prompt2"]},
                                                      return_exceptions=True)) == ["prompt1", "prompt2"]

        # Test with return_exceptions=False
        with pytest.raises(RuntimeError, match="unittest"):
            await client.generate_batch_async({'prompt': ["prompt1", "error", "prompt3"]}, return_exceptions=False)


async def test_generate_batch_async_error():
    with mock.patch("langchain_nvidia_ai_endpoints.ChatNVIDIA.agenerate_prompt", autospec=True) as mock_nvfoundationllm:

        def mock_generation_side_effect(*_, **kwargs):
            raise RuntimeError("unittest")

        mock_nvfoundationllm.side_effect = mock_generation_side_effect

        client = NVFoundationLLMService().get_client(model_name="test_model")

        with pytest.raises(RuntimeError, match="unittest"):
            await client.generate_batch_async({'prompt': ["prompt1", "prompt2"]}, return_exceptions=False)
