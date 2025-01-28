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

import numpy as np
import pytest
import typing_utils

from morpheus.common import FilterSource
from morpheus.messages import ControlMessage
from morpheus.messages import TensorMemory
from morpheus.messages.message_meta import MessageMeta
from morpheus.stages.postprocess.filter_detections_stage import FilterDetectionsStage


def _make_control_message(df, probs):
    df_ = df[0:len(probs)]
    cm = ControlMessage()
    cm.payload(MessageMeta(df_))
    cm.tensors(TensorMemory(count=len(df_), tensors={'probs': probs}))

    return cm


def test_constructor(config):
    fds = FilterDetectionsStage(config)
    assert fds.name == "filter"

    # Just ensure that we get a valid non-empty tuple
    accepted_union = typing.Union[fds.accepted_types()]
    assert typing_utils.issubtype(ControlMessage, accepted_union)


@pytest.mark.use_pandas
def test_filter_copy(config, filter_probs_df):
    fds = FilterDetectionsStage(config, threshold=0.5, filter_source=FilterSource.TENSOR)

    probs = np.array([[0.1, 0.5, 0.3], [0.2, 0.3, 0.4]])
    mock_control_message = _make_control_message(filter_probs_df, probs)

    # All values are at or below the threshold so nothing should be returned
    output_control_message = fds._controller.filter_copy(mock_control_message)
    assert output_control_message is None

    # Only one row has a value above the threshold
    probs = np.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.5, 0.8],
        [0.2, 0.4, 0.3],
    ])

    mock_control_message = _make_control_message(filter_probs_df, probs)
    output_control_message = fds._controller.filter_copy(mock_control_message)
    assert output_control_message.payload().get_data().to_numpy().tolist() == filter_probs_df.loc[
        1:1, :].to_numpy().tolist()

    # Two adjacent rows have a value above the threashold
    probs = np.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.2, 0.3],
        [0.1, 0.5, 0.8],
        [0.1, 0.9, 0.2],
        [0.2, 0.4, 0.3],
    ])

    mock_control_message = _make_control_message(filter_probs_df, probs)
    output_control_message = fds._controller.filter_copy(mock_control_message)
    assert output_control_message.payload().get_data().to_numpy().tolist() == filter_probs_df.loc[
        2:3, :].to_numpy().tolist()

    # Two non-adjacent rows have a value above the threashold
    probs = np.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.2, 0.3],
        [0.1, 0.5, 0.8],
        [0.4, 0.3, 0.2],
        [0.1, 0.9, 0.2],
        [0.2, 0.4, 0.3],
    ])

    mask = np.zeros(len(filter_probs_df), dtype=np.bool_)
    mask[2] = True
    mask[4] = True

    mock_control_message = _make_control_message(filter_probs_df, probs)
    output_control_message = fds._controller.filter_copy(mock_control_message)
    assert output_control_message.payload().get_data().to_numpy().tolist() == filter_probs_df.loc[
        mask, :].to_numpy().tolist()


@pytest.mark.use_pandas
@pytest.mark.parametrize('do_copy', [True, False])
@pytest.mark.parametrize('threshold', [0.1, 0.5, 0.8])
@pytest.mark.parametrize('field_name', ['v1', 'v2', 'v3', 'v4'])
def test_filter_column(config, filter_probs_df, do_copy, threshold, field_name):
    fds = FilterDetectionsStage(config,
                                threshold=threshold,
                                copy=do_copy,
                                filter_source=FilterSource.DATAFRAME,
                                field_name=field_name)
    expected_df = filter_probs_df[filter_probs_df[field_name] > threshold]

    probs = np.zeros([len(filter_probs_df), 3], 'float')
    mock_control_message = _make_control_message(filter_probs_df, probs)
    output_control_message = fds._controller.filter_copy(mock_control_message)
    assert output_control_message.payload().get_data().to_numpy().tolist() == expected_df.to_numpy().tolist()


@pytest.mark.use_pandas
def test_filter_slice(config, filter_probs_df):
    fds = FilterDetectionsStage(config, threshold=0.5, filter_source=FilterSource.TENSOR)

    probs = np.array([[0.1, 0.5, 0.3], [0.2, 0.3, 0.4]])

    # All values are at or below the threshold

    mock_control_message = _make_control_message(filter_probs_df, probs)
    output_control_message = fds._controller.filter_slice(mock_control_message)
    assert len(output_control_message) == 0

    # Only one row has a value above the threshold
    probs = np.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.5, 0.8],
        [0.2, 0.4, 0.3],
    ])

    mock_control_message = _make_control_message(filter_probs_df, probs)
    output_control_message = fds._controller.filter_slice(mock_control_message)
    assert output_control_message[0].payload().get_data().to_numpy().tolist() == filter_probs_df.loc[
        1:1, :].to_numpy().tolist()

    # Two adjacent rows have a value above the threashold
    probs = np.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.2, 0.3],
        [0.1, 0.5, 0.8],
        [0.1, 0.9, 0.2],
        [0.2, 0.4, 0.3],
    ])

    mock_control_message = _make_control_message(filter_probs_df, probs)
    output_control_message = fds._controller.filter_slice(mock_control_message)
    assert output_control_message[0].payload().get_data().to_numpy().tolist() == filter_probs_df.loc[
        2:3, :].to_numpy().tolist()

    # Two non-adjacent rows have a value above the threashold
    probs = np.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.2, 0.3],
        [0.1, 0.5, 0.8],
        [0.4, 0.3, 0.2],
        [0.1, 0.9, 0.2],
        [0.2, 0.4, 0.3],
    ])

    mock_control_message = _make_control_message(filter_probs_df, probs)
    output_control_message = fds._controller.filter_slice(mock_control_message)
    assert len(output_control_message) == 2
    (control_msg1, control_msg2) = output_control_message  # pylint: disable=unbalanced-tuple-unpacking
    assert control_msg1.payload().count == 1
    assert control_msg2.payload().count == 1

    assert control_msg1.payload().get_data().to_numpy().tolist() == filter_probs_df.loc[2:2, :].to_numpy().tolist()
    assert control_msg2.payload().get_data().to_numpy().tolist() == filter_probs_df.loc[4:4, :].to_numpy().tolist()
