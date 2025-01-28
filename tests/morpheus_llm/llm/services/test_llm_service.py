# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import inspect
import os
from abc import ABC
from unittest import mock

import pytest

from morpheus_llm.llm.services.llm_service import LLMClient
from morpheus_llm.llm.services.llm_service import LLMService
from morpheus_llm.llm.services.nemo_llm_service import NeMoLLMService
from morpheus_llm.llm.services.nvfoundation_llm_service import NVFoundationLLMService
from morpheus_llm.llm.services.openai_chat_service import OpenAIChatService


@pytest.mark.parametrize("cls", [LLMClient, LLMService])
def test_is_abstract(cls: ABC):
    assert inspect.isabstract(cls)


@pytest.mark.usefixtures("restore_environ")
@pytest.mark.parametrize("service_name, expected_cls, env_values",
                         [("nemo", NeMoLLMService, {}), ("openai", OpenAIChatService, {
                             'OPENAI_API_KEY': 'test_api'
                         }),
                          pytest.param("nvfoundation", NVFoundationLLMService, {'NVIDIA_API_KEY': 'test_api'})])
def test_create(service_name: str, expected_cls: type, env_values: dict[str, str]):
    if env_values:
        os.environ.update(env_values)

    service = LLMService.create(service_name)
    assert isinstance(service, expected_cls)


@pytest.mark.parametrize(
    "service_name, class_name",
    [("nemo", "morpheus_llm.llm.services.nemo_llm_service.NeMoLLMService"),
     ("openai", "morpheus_llm.llm.services.openai_chat_service.OpenAIChatService"),
     ("nvfoundation", "morpheus_llm.llm.services.nvfoundation_llm_service.NVFoundationLLMService")])
def test_create_mocked(service_name: str, class_name: str):
    with mock.patch(class_name) as mock_cls:
        mock_instance = mock.MagicMock()
        mock_cls.return_value = mock_instance

        service = LLMService.create(service_name)
        mock_cls.assert_called_once()
        assert service is mock_instance
