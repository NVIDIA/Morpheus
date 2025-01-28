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

import os
from unittest import mock

import pytest

from morpheus.utils.env_config_value import EnvConfigValue
from morpheus.utils.env_config_value import EnvConfigValueSource


class EnvDrivenValue(EnvConfigValue):
    _ENV_KEY = "DEFAULT"
    _ENV_KEY_OVERRIDE = "OVERRIDE"


def test_env_driven_value():
    with mock.patch.dict(os.environ, clear=True, values={"DEFAULT": "default.api.com"}):

        config = EnvDrivenValue()
        assert config.value == "default.api.com"
        assert config.source == EnvConfigValueSource.ENV_DEFAULT
        assert config.use_env

        with pytest.raises(ValueError):
            config = EnvDrivenValue(use_env=False)

        config = EnvDrivenValue("api.com")
        assert config.value == "api.com"
        assert config.source == EnvConfigValueSource.CONSTRUCTOR
        assert config.use_env

    with mock.patch.dict(os.environ, clear=True, values={"OVERRIDE": "override.api.com"}):

        config = EnvDrivenValue("api.com")
        assert config.value == "override.api.com"
        assert config.source == EnvConfigValueSource.ENV_OVERRIDE
        assert config.use_env

        config = EnvDrivenValue("api.com", use_env=False)
        assert config.value == "api.com"
        assert config.source == EnvConfigValueSource.CONSTRUCTOR
        assert not config.use_env


class EnvDriverValueNoOverride(EnvConfigValue):
    _ENV_KEY = "DEFAULT"


def test_env_driven_value_no_override():
    with mock.patch.dict(os.environ, clear=True, values={"DEFAULT": "default.api.com"}):

        config = EnvDriverValueNoOverride()
        assert config.value == "default.api.com"
        assert config.source == EnvConfigValueSource.ENV_DEFAULT
        assert config.use_env

        with pytest.raises(ValueError):
            config = EnvDriverValueNoOverride(use_env=False)

        config = EnvDriverValueNoOverride("api.com")
        assert config.value == "api.com"
        assert config.source == EnvConfigValueSource.CONSTRUCTOR
        assert config.use_env

    with mock.patch.dict(os.environ, clear=True, values={"OVERRIDE": "override.api.com"}):

        config = EnvDriverValueNoOverride("api.com")
        assert config.value == "api.com"
        assert config.source == EnvConfigValueSource.CONSTRUCTOR
        assert config.use_env


class EnvDrivenValueNoDefault(EnvConfigValue):
    _ENV_KEY_OVERRIDE = "OVERRIDE"


def test_env_driven_value_no_default():
    with mock.patch.dict(os.environ, clear=True, values={"DEFAULT": "default.api.com"}):

        with pytest.raises(ValueError):
            config = EnvDrivenValueNoDefault()

        config = EnvDrivenValueNoDefault("api.com")
        assert config.value == "api.com"
        assert config.source == EnvConfigValueSource.CONSTRUCTOR
        assert config.use_env

    with mock.patch.dict(os.environ, clear=True, values={"OVERRIDE": "override.api.com"}):

        config = EnvDrivenValueNoDefault("api.com")
        assert config.value == "override.api.com"
        assert config.source == EnvConfigValueSource.ENV_OVERRIDE
        assert config.use_env


class EnvOptionalValue(EnvConfigValue):
    _ALLOW_NONE = True


def test_env_optional_value():
    config = EnvOptionalValue()
    assert config.value is None
    assert config.source == EnvConfigValueSource.CONSTRUCTOR
    assert config.use_env
