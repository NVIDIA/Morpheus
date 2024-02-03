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


@pytest.fixture(name="mock_llm_client")
def mock_llm_client_fixture():
    from morpheus.llm.services.llm_service import LLMClient
    mock_client = mock.MagicMock(LLMClient)
    mock_client.return_value = mock_client
    mock_client.get_input_names.return_value = ["prompt"]
    mock_client.generate_batch_async = mock.AsyncMock()
    return mock_client


@pytest.fixture(name="mock_langchain_agent_executor")
def mock_langchain_agent_executor_fixture():
    mock_agent_ex = mock.MagicMock()
    mock_agent_ex.return_value = mock_agent_ex
    mock_agent_ex.input_keys = ["prompt"]
    mock_agent_ex.arun = mock.AsyncMock()
    return mock_agent_ex


@pytest.fixture(name="mock_haystack_agent")
def mock_haystack_agent_fixture():
    mock_agent = mock.MagicMock()
    mock_agent.return_value = mock_agent
    mock_agent.run = mock.Mock()
    return mock_agent


@pytest.fixture(name="mock_haystack_answer_data", scope="module")
def mock_haystack_answer_data_fixture() -> dict:
    return {
        'answer': '4.95',
        'type': 'generative',
        'score': None,
        'context': None,
        'offsets_in_document': None,
        'offsets_in_context': None,
        'document_ids': None,
        'meta': {}
    }


@pytest.fixture(name="mock_haystack_agent_run_return", scope="module")
def mock_haystack_agent_run_return_fixture(mock_haystack_answer_data: mock.MagicMock) -> dict:
    from haystack.schema import Answer

    answer = Answer(**mock_haystack_answer_data)
    return {'query': {'query': 'test query'}, 'answers': [answer], 'transcript': 'query transcript'}
