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

from morpheus.llm.services.llm_service import LLMService
from morpheus.llm.services.openai_chat_service import OpenAIChatClient
from morpheus.llm.services.openai_chat_service import OpenAIChatService


def test_constructor():
    service = OpenAIChatService()
    assert isinstance(service, LLMService)


def test_get_client():
    service = OpenAIChatService()
    client = service.get_client("test_model")

    assert isinstance(client, OpenAIChatClient)


@pytest.mark.parametrize("set_assistant", [True, False])
@pytest.mark.parametrize("temperature", [0, 1, 2])
@mock.patch("morpheus.llm.services.openai_chat_service.OpenAIChatClient")
def test_get_client_passed_args(mock_client: mock.MagicMock, set_assistant: bool, temperature: int):
    service = OpenAIChatService()
    service.get_client("test_model", set_assistant=set_assistant, temperature=temperature, test='this')

    # Ensure the get_client method passed on the set_assistant and model kwargs
    mock_client.assert_called_once_with(service,
                                        model_name="test_model",
                                        set_assistant=set_assistant,
                                        temperature=temperature,
                                        test='this')
