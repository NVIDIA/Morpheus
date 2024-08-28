#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cupy as cp
import pytest

from morpheus.common import FilterSource
from morpheus.messages import ControlMessage
from morpheus.messages import MultiResponseMessage
from morpheus.messages import ResponseMemory
from morpheus.messages.message_meta import MessageMeta
from morpheus.stages.postprocess.filter_detections_stage import FilterDetectionsStage


def _make_multi_response_message(df, probs):
    df_ = df[0:len(probs)]
    mem = ResponseMemory(count=len(df_), tensors={'probs': probs})

    return MultiResponseMessage(meta=MessageMeta(df_), memory=mem)


def _make_control_message(df, probs):
    df_ = df[0:len(probs)]
    cm = ControlMessage()
    cm.payload(MessageMeta(df_))
    cm.tensors(count=len(df_), tensors={'probs': probs})

    return cm


def test_constructor(config):
    fds = FilterDetectionsStage(config)
    assert fds.name == "filter"

    # Just ensure that we get a valid non-empty tuple
    accepted_types = fds.accepted_types()
    assert isinstance(accepted_types, tuple)
    assert len(accepted_types) > 0


@pytest.mark.use_cudf
def test_filter_copy(config, filter_probs_df):
    fds = FilterDetectionsStage(config, threshold=0.5, filter_source=FilterSource.TENSOR)

    probs = cp.array([[0.1, 0.5, 0.3], [0.2, 0.3, 0.4]])
    mock_multi_response_message = _make_multi_response_message(filter_probs_df, probs)
    mock_control_message = _make_control_message(filter_probs_df, probs)

    # All values are at or below the threshold so nothing should be returned
    output_multi_response_message = fds._controller.filter_copy(mock_multi_response_message)
    assert output_multi_response_message is None
    output_control_message = fds._controller.filter_copy(mock_control_message)
    assert output_control_message is None

    # Only one row has a value above the threshold
    probs = cp.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.5, 0.8],
        [0.2, 0.4, 0.3],
    ])

    mock_multi_response_message = _make_multi_response_message(filter_probs_df, probs)
    output_multi_response_message = fds._controller.filter_copy(mock_multi_response_message)
    assert output_multi_response_message.get_meta().to_cupy().tolist() == filter_probs_df.loc[1:1, :].to_cupy().tolist()
    mock_control_message = _make_control_message(filter_probs_df, probs)
    output_control_message = fds._controller.filter_copy(mock_control_message)
    assert output_control_message.payload().get_data().to_cupy().tolist() == output_multi_response_message.get_meta(
    ).to_cupy().tolist()

    # Two adjacent rows have a value above the threashold
    probs = cp.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.2, 0.3],
        [0.1, 0.5, 0.8],
        [0.1, 0.9, 0.2],
        [0.2, 0.4, 0.3],
    ])

    mock_multi_response_message = _make_multi_response_message(filter_probs_df, probs)
    output_multi_response_message = fds._controller.filter_copy(mock_multi_response_message)
    assert output_multi_response_message.get_meta().to_cupy().tolist() == filter_probs_df.loc[2:3, :].to_cupy().tolist()
    mock_control_message = _make_control_message(filter_probs_df, probs)
    output_control_message = fds._controller.filter_copy(mock_control_message)
    assert output_control_message.payload().get_data().to_cupy().tolist() == output_multi_response_message.get_meta(
    ).to_cupy().tolist()

    # Two non-adjacent rows have a value above the threashold
    probs = cp.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.2, 0.3],
        [0.1, 0.5, 0.8],
        [0.4, 0.3, 0.2],
        [0.1, 0.9, 0.2],
        [0.2, 0.4, 0.3],
    ])

    mask = cp.zeros(len(filter_probs_df), dtype=cp.bool_)
    mask[2] = True
    mask[4] = True

    mock_multi_response_message = _make_multi_response_message(filter_probs_df, probs)
    output_multi_response_message = fds._controller.filter_copy(mock_multi_response_message)
    assert output_multi_response_message.get_meta().to_cupy().tolist() == filter_probs_df.loc[
        mask, :].to_cupy().tolist()
    mock_control_message = _make_control_message(filter_probs_df, probs)
    output_control_message = fds._controller.filter_copy(mock_control_message)
    assert output_control_message.payload().get_data().to_cupy().tolist() == output_multi_response_message.get_meta(
    ).to_cupy().tolist()


@pytest.mark.use_cudf
@pytest.mark.parametrize('do_copy', [True, False])
@pytest.mark.parametrize('threshold', [0.1, 0.5, 0.8])
@pytest.mark.parametrize('field_name', ['v1', 'v2', 'v3', 'v4'])
def test_filter_column(config, filter_probs_df, do_copy, threshold, field_name):
    fds = FilterDetectionsStage(config,
                                threshold=threshold,
                                copy=do_copy,
                                filter_source=FilterSource.DATAFRAME,
                                field_name=field_name)
    expected_df = filter_probs_df.to_pandas()
    expected_df = expected_df[expected_df[field_name] > threshold]

    probs = cp.zeros([len(filter_probs_df), 3], 'float')
    mock_multi_response_message = _make_multi_response_message(filter_probs_df, probs)
    # All values are at or below the threshold
    output_multi_response_message = fds._controller.filter_copy(mock_multi_response_message)
    assert output_multi_response_message.get_meta().to_cupy().tolist() == expected_df.to_numpy().tolist()
    mock_control_message = _make_control_message(filter_probs_df, probs)
    output_control_message = fds._controller.filter_copy(mock_control_message)
    assert output_control_message.payload().get_data().to_cupy().tolist() == output_multi_response_message.get_meta(
    ).to_cupy().tolist()


@pytest.mark.use_cudf
def test_filter_slice(config, filter_probs_df):
    fds = FilterDetectionsStage(config, threshold=0.5, filter_source=FilterSource.TENSOR)

    probs = cp.array([[0.1, 0.5, 0.3], [0.2, 0.3, 0.4]])
    mock_multi_response_message = _make_multi_response_message(filter_probs_df, probs)

    # All values are at or below the threshold
    output_multi_response_messages = fds._controller.filter_slice(mock_multi_response_message)
    assert len(output_multi_response_messages) == 0
    mock_control_message = _make_control_message(filter_probs_df, probs)
    output_control_message = fds._controller.filter_slice(mock_control_message)
    assert len(output_control_message) == len(output_multi_response_messages)

    # Only one row has a value above the threshold
    probs = cp.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.5, 0.8],
        [0.2, 0.4, 0.3],
    ])

    mock_multi_response_message: MultiResponseMessage = _make_multi_response_message(filter_probs_df, probs)

    output_multi_response_messages = fds._controller.filter_slice(mock_multi_response_message)
    assert len(output_multi_response_messages) == 1
    assert output_multi_response_messages[0].get_meta().to_cupy().tolist() == filter_probs_df.loc[
        1:1, :].to_cupy().tolist()

    mock_control_message = _make_control_message(filter_probs_df, probs)
    output_control_message = fds._controller.filter_slice(mock_control_message)
    assert len(output_control_message) == len(output_multi_response_messages)
    assert output_control_message[0].payload().get_data().to_cupy().tolist(
    ) == output_multi_response_messages[0].get_meta().to_cupy().tolist()

    # Two adjacent rows have a value above the threashold
    probs = cp.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.2, 0.3],
        [0.1, 0.5, 0.8],
        [0.1, 0.9, 0.2],
        [0.2, 0.4, 0.3],
    ])

    mock_multi_response_message = _make_multi_response_message(filter_probs_df, probs)

    output_multi_response_messages = fds._controller.filter_slice(mock_multi_response_message)
    assert len(output_multi_response_messages) == 1
    assert output_multi_response_messages[0].offset == 2
    assert output_multi_response_messages[0].count == 2
    assert output_multi_response_messages[0].get_meta().to_cupy().tolist() == filter_probs_df.loc[
        2:3, :].to_cupy().tolist()

    mock_control_message = _make_control_message(filter_probs_df, probs)
    output_control_message = fds._controller.filter_slice(mock_control_message)
    assert len(output_control_message) == len(output_multi_response_messages)
    assert output_control_message[0].payload().get_data().to_cupy().tolist(
    ) == output_multi_response_messages[0].get_meta().to_cupy().tolist()

    # Two non-adjacent rows have a value above the threashold
    probs = cp.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.2, 0.3],
        [0.1, 0.5, 0.8],
        [0.4, 0.3, 0.2],
        [0.1, 0.9, 0.2],
        [0.2, 0.4, 0.3],
    ])

    mock_multi_response_message = _make_multi_response_message(filter_probs_df, probs)

    output_multi_response_messages = fds._controller.filter_slice(mock_multi_response_message)
    assert len(output_multi_response_messages) == 2

    # pylint: disable=unbalanced-tuple-unpacking
    (multi_response_msg1, multi_response_msg2) = output_multi_response_messages
    assert multi_response_msg1.offset == 2
    assert multi_response_msg1.count == 1

    assert multi_response_msg2.offset == 4
    assert multi_response_msg2.count == 1

    assert multi_response_msg1.get_meta().to_cupy().tolist() == filter_probs_df.loc[2:2, :].to_cupy().tolist()
    assert multi_response_msg2.get_meta().to_cupy().tolist() == filter_probs_df.loc[4:4, :].to_cupy().tolist()

    mock_control_message = _make_control_message(filter_probs_df, probs)
    output_control_message = fds._controller.filter_slice(mock_control_message)
    assert len(output_control_message) == len(output_multi_response_messages)
    (control_msg1, control_msg2) = output_control_message  # pylint: disable=unbalanced-tuple-unpacking
    assert control_msg1.payload().count == multi_response_msg1.count
    assert control_msg2.payload().count == multi_response_msg2.count

    assert control_msg1.payload().get_data().to_cupy().tolist() == multi_response_msg1.get_meta().to_cupy().tolist()
    assert control_msg2.payload().get_data().to_cupy().tolist() == multi_response_msg2.get_meta().to_cupy().tolist()
