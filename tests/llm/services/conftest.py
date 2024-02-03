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

# Override fixtures from parent setting autouse to True


@pytest.fixture(name="nemollm", autouse=True, scope='session')
def nemollm_fixture(nemollm):
    """
    All of the tests in this subdir require nemollm
    """
    yield nemollm


@pytest.fixture(name="openai", autouse=True, scope='session')
def openai_fixture(openai):
    """
    All of the tests in this subdir require openai
    """
    yield openai


@pytest.mark.usefixtures("nemollm")
@pytest.fixture(name="mock_nemollm", autouse=True)
def mock_nemollm_fixture(mock_nemollm):
    yield mock_nemollm


@pytest.fixture(name="mock_nemo_service")
def mock_nemo_service_fixture(mock_nemollm: mock.MagicMock):
    mock_nemo_service = mock.MagicMock()
    mock_nemo_service.return_value = mock_nemo_service
    mock_nemo_service._conn = mock_nemollm
    return mock_nemo_service
