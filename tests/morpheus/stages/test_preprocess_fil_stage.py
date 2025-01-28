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

import typing

import pytest
import typing_utils

from morpheus.config import Config
from morpheus.config import ConfigFIL
from morpheus.messages import ControlMessage
from morpheus.stages.preprocess.preprocess_fil_stage import PreprocessFILStage


@pytest.fixture(name='config')
def fixture_config(config: Config):
    config.feature_length = 1
    config.fil = ConfigFIL()
    config.fil.feature_columns = ["data"]
    yield config


def test_constructor(config: Config):
    stage = PreprocessFILStage(config)
    assert stage.name == "preprocess-fil"
    assert stage._fea_length == config.feature_length
    assert stage.features == config.fil.feature_columns

    accepted_union = typing.Union[stage.accepted_types()]
    assert typing_utils.issubtype(ControlMessage, accepted_union)
