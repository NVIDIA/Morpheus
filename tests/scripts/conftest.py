# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import types

import pytest


@pytest.fixture(name="morpheus_llm", scope='session', autouse=True)
def morpheus_llm_fixture(morpheus_llm: types.ModuleType):
    """
    Fixture to ensure morpheus_llm is installed
    """
    yield morpheus_llm


@pytest.fixture(name="morpheus_dfp", scope='session', autouse=True)
def morpheus_dfp_fixture(morpheus_dfp: types.ModuleType):
    """
    Fixture to ensure morpheus_dfp is installed
    """
    yield morpheus_dfp
