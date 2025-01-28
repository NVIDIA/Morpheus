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
from unittest.mock import patch

import cupy as cp
import pytest
import typing_utils

from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.messages import TensorMemory
from morpheus.stages.postprocess.ml_flow_drift_stage import MLFlowDriftStage


def _make_control_message(df, probs):
    df_ = df[0:len(probs)]
    cm = ControlMessage()
    cm.payload(MessageMeta(df_))
    cm.tensors(TensorMemory(count=len(df_), tensors={'probs': probs}))

    return cm


def test_constructor(config):
    with patch("morpheus.stages.postprocess.ml_flow_drift_stage.mlflow.start_run"):
        stage = MLFlowDriftStage(config)
    assert stage.name == "mlflow_drift"

    accepted_union = typing.Union[stage.accepted_types()]
    assert typing_utils.issubtype(ControlMessage, accepted_union)


@pytest.mark.use_cudf
def test_calc_drift(config, filter_probs_df):
    with patch("morpheus.stages.postprocess.ml_flow_drift_stage.mlflow.start_run"):
        labels = ["a", "b", "c"]
        stage = MLFlowDriftStage(config, labels=labels, batch_size=1)

    probs = cp.array([[0.1, 0.5, 0.3], [0.2, 0.3, 0.4]])

    mock_control_message = _make_control_message(filter_probs_df, probs)

    expected_metrics = [{
        'a': 0.9, 'b': 0.5, 'c': 0.7, 'total': 0.6999999999999998
    }, {
        'a': 0.8, 'b': 0.7, 'c': 0.6, 'total': 0.7000000000000001
    }]

    control_message_metrics = []
    with patch("morpheus.stages.postprocess.ml_flow_drift_stage.mlflow.log_metrics") as mock_log_metrics:
        stage._calc_drift(mock_control_message)
        for call_arg in mock_log_metrics.call_args_list:
            control_message_metrics.append(call_arg[0][0])
    assert control_message_metrics == expected_metrics
