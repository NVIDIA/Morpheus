#!/usr/bin/env python
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

import typing

import cupy as cp
import pandas as pd
import pytest
import typing_utils

from morpheus.config import Config
from morpheus.config import ConfigAutoEncoder
from morpheus.messages import ControlMessage
from morpheus.messages import TensorMemory
from morpheus.messages.message_meta import MessageMeta
from morpheus.stages.postprocess.timeseries_stage import TimeSeriesStage


@pytest.fixture(name='config')
def fixture_config(config: Config):
    config.feature_length = 256
    config.ae = ConfigAutoEncoder()
    config.ae.feature_columns = ["data"]
    config.ae.timestamp_column_name = "ts"
    yield config


def _make_control_message(df, probs):
    df_ = df[0:len(probs)]
    cm = ControlMessage()
    cm.payload(MessageMeta(df_))
    cm.tensors(TensorMemory(count=len(df_), tensors={'probs': probs}))
    cm.set_metadata("user_id", "test_user_id")

    return cm


def test_constructor(config):
    stage = TimeSeriesStage(config)
    assert stage.name == "timeseries"

    accepted_union = typing.Union[stage.accepted_types()]
    assert typing_utils.issubtype(ControlMessage, accepted_union)


@pytest.mark.cpu_mode
def test_call_timeseries_user(config):
    stage = TimeSeriesStage(config)

    df = pd.DataFrame({"ts": pd.date_range(start='01-01-2022', periods=5)})
    probs = cp.array([[0.1, 0.5, 0.3], [0.2, 0.3, 0.4]])
    mock_control_message = _make_control_message(df, probs)

    assert stage._call_timeseries_user(mock_control_message)[0].get_metadata("user_id") == "test_user_id"
