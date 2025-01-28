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

from morpheus_llm.llm.services.nemo_llm_service import NeMoLLMClient
from morpheus_llm.llm.services.nemo_llm_service import NeMoLLMService


@pytest.mark.usefixtures("restore_environ")
@pytest.mark.parametrize("api_key", [None, "test_api_key"])
@pytest.mark.parametrize("org_id", [None, "test_org_id"])
def test_constructor(mock_nemollm: mock.MagicMock, api_key: str, org_id: str):
    """
    Test that the constructor prefers explicit arguments over environment variables.
    """
    env_api_key = "test_env_api_key"
    env_org_id = "test_env_org_id"
    os.environ["NGC_API_KEY"] = env_api_key
    os.environ["NGC_ORG_ID"] = env_org_id

    expected_api_key = api_key or env_api_key
    expected_org_id = org_id or env_org_id

    NeMoLLMService(api_key=api_key, org_id=org_id)
    _, kwargs = mock_nemollm.call_args_list[-1]

    assert kwargs["api_key"] == expected_api_key
    assert kwargs["org_id"] == expected_org_id


def test_get_client():
    service = NeMoLLMService(api_key="test_api_key")
    client = service.get_client(model_name="test_model")

    assert isinstance(client, NeMoLLMClient)
