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

from unittest import mock

import pytest

from morpheus_llm.llm.services.llm_service import LLMClient
from morpheus_llm.llm.services.nemo_llm_service import NeMoLLMService


def test_constructor():
    client = NeMoLLMService(api_key="dummy").get_client(model_name="test_model", additional_arg="test_arg")

    assert isinstance(client, LLMClient)


def test_get_input_names():
    client = NeMoLLMService(api_key="dummy").get_client(model_name="test_model", additional_arg="test_arg")

    assert client.get_input_names() == ["prompt"]


def test_generate(mock_nemollm: mock.MagicMock):

    client = NeMoLLMService(api_key="dummy").get_client(model_name="test_model", customization_id="test_custom_id")

    assert client.generate(prompt="test_prompt") == "test_prompt"

    mock_nemollm.generate_multiple.assert_called_once_with(model="test_model",
                                                           prompts=["test_prompt"],
                                                           return_type="text",
                                                           customization_id="test_custom_id")


def test_generate_batch(mock_nemollm: mock.MagicMock):

    client = NeMoLLMService(api_key="dummy").get_client(model_name="test_model", customization_id="test_custom_id")

    assert client.generate_batch({'prompt': ["prompt1", "prompt2"]}) == ["prompt1", "prompt2"]

    mock_nemollm.generate_multiple.assert_called_once_with(model="test_model",
                                                           prompts=["prompt1", "prompt2"],
                                                           return_type="text",
                                                           customization_id="test_custom_id")


def test_generate_batch_exception(mock_nemollm: mock.MagicMock):

    def mock_post_process_generate_response(*args, **_):
        if ("error" in args[0]):
            raise RuntimeError("unittest")

        if ("fail" in args[0]):
            return {"status": "fail", "msg": "unittest"}

        return {"status": "success", "text": args[0]}

    mock_nemollm.post_process_generate_response.side_effect = mock_post_process_generate_response

    client = NeMoLLMService(api_key="dummy").get_client(model_name="test_model", customization_id="test_custom_id")

    # Test warning, return_exceptions=True
    with pytest.warns(UserWarning):
        assert client.generate_batch({'prompt': ["prompt1", "prompt2", "prompt3"]},
                                     return_exceptions=True) == ["prompt1", "prompt2", "prompt3"]

    # Test failures, return_exceptions=False
    with pytest.raises(RuntimeError, match="unittest"):
        client.generate_batch({'prompt': ["prompt1", "error", "prompt3"]}, return_exceptions=False)

    # Test exceptions, return_exceptions=False
    with pytest.raises(RuntimeError, match="unittest"):
        client.generate_batch({'prompt': ["prompt1", "error", "prompt3"]}, return_exceptions=False)


async def test_generate_async(mock_nemollm: mock.MagicMock):

    client = NeMoLLMService(api_key="dummy").get_client(model_name="test_model", customization_id="test_custom_id")

    results = await client.generate_async(prompt="test_prompt")

    assert results == "test_output"

    mock_nemollm.generate.assert_called_once_with("test_model",
                                                  "test_prompt",
                                                  return_type="async",
                                                  customization_id="test_custom_id")


async def test_generate_batch_async(mock_nemollm: mock.MagicMock):
    # mock_nemollm.post_process_generate_response.side_effect = [{"text": "output1"}, {"text": "output2"}]

    client = NeMoLLMService(api_key="dummy").get_client(model_name="test_model", customization_id="test_custom_id")

    results = await client.generate_batch_async({'prompt': ["prompt1", "prompt2"]})

    assert results == ["test_output", "test_output"]

    mock_nemollm.generate.assert_has_calls([
        mock.call("test_model", "prompt1", return_type="async", customization_id="test_custom_id"),
        mock.call("test_model", "prompt2", return_type="async", customization_id="test_custom_id")
    ])


async def test_generate_batch_async_exception(mock_nemollm: mock.MagicMock):

    def mock_post_process_generate_response(*args, **_):
        if ("error" in args[0]):
            raise RuntimeError("unittest")

        if ("fail" in args[0]):
            return {"status": "fail", "msg": "unittest"}

        return {"status": "success", "text": args[0]}

    mock_nemollm.post_process_generate_response.side_effect = mock_post_process_generate_response

    client = NeMoLLMService(api_key="dummy").get_client(model_name="test_model", customization_id="test_custom_id")

    # Test failures, return_exceptions=True
    results = await client.generate_batch_async({'prompt': ["prompt1", "fail", "prompt3"]}, return_exceptions=True)
    assert isinstance(results[1], RuntimeError)

    # Test failures, return_exceptions=False
    with pytest.raises(RuntimeError, match="unittest"):
        await client.generate_batch_async({'prompt': ["prompt1", "fail", "prompt3"]}, return_exceptions=False)

    # Test exceptions, return_exceptions=True
    results = await client.generate_batch_async({'prompt': ["prompt1", "error", "prompt3"]}, return_exceptions=True)
    assert isinstance(results[1], RuntimeError)

    # Test exceptions, return_exceptions=False
    with pytest.raises(RuntimeError, match="unittest"):
        await client.generate_batch_async({'prompt': ["prompt1", "error", "prompt3"]}, return_exceptions=False)


async def test_generate_batch_async_error_retry(mock_nemollm: mock.MagicMock):

    count = 0

    def mock_post_process_generate_response(*args, **_):
        nonlocal count
        if count < 2:
            count += 1
            return {"status": "fail", "msg": "unittest"}
        return {"status": "success", "text": args[0]}

    mock_nemollm.post_process_generate_response.side_effect = mock_post_process_generate_response

    client = NeMoLLMService(api_key="dummy", retry_count=2).get_client(model_name="test_model",
                                                                       customization_id="test_custom_id")

    results = await client.generate_batch_async({'prompt': ["prompt1", "prompt2"]})

    assert results == ["prompt1", "prompt2"]
