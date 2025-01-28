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

import cupy as cp
import pandas as pd
import typing_utils

import cudf

from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.messages import TensorMemory
from morpheus.stages.postprocess.generate_viz_frames_stage import GenerateVizFramesStage


def _make_control_message(df, probs):
    df_ = df[0:len(probs)]
    cm = ControlMessage()
    cm.payload(MessageMeta(df_))
    cm.tensors(TensorMemory(count=len(df_), tensors={'probs': probs}))

    return cm


def test_constructor(config: Config):
    stage = GenerateVizFramesStage(config)
    assert stage.name == "gen_viz"

    accepted_union = typing.Union[stage.accepted_types()]
    assert typing_utils.issubtype(ControlMessage, accepted_union)


def test_process_control_message(config: Config):
    stage = GenerateVizFramesStage(config)

    df = cudf.DataFrame({
        "timestamp": [1616380971990, 1616380971991],
        "src_ip": ["10.20.16.248", "10.244.0.1"],
        "dest_ip": ["10.244.0.59", "10.244.0.25"],
        "src_port": ["50410", "50410"],
        "dest_port": ["80", "80"],
        "data": ["a", "b"]
    })

    probs = cp.array([[0.1, 0.5, 0.3], [0.2, 0.3, 0.4]])
    mock_control_message = _make_control_message(df, probs)
    output_control_message_list = stage._to_vis_df(mock_control_message)

    expected_df = pd.DataFrame({
        'timestamp': [1616380971990, 1616380971991],
        'src_ip': ['10.20.16.248', '10.244.0.1'],
        'dest_ip': ['10.244.0.59', '10.244.0.25'],
        'src_port': ['50410', '50410'],
        'dest_port': ['80', '80'],
        'data': ['a', 'b'],
        'si': ['bank_acct', 'none'],
        'ts_round_sec': [1616380971000, 1616380971000]
    })

    for output_control_message in output_control_message_list:
        assert output_control_message[1].equals(expected_df)
