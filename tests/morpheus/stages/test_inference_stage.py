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

import asyncio
from unittest import mock

import cupy as cp
import pytest

import cudf

from _utils.inference_worker import IW
from morpheus.messages import ControlMessage
from morpheus.messages import InferenceMemory
from morpheus.messages import ResponseMemory
from morpheus.messages.message_meta import MessageMeta
from morpheus.stages.inference.inference_stage import InferenceStage


class InferenceStageT(InferenceStage):
    # Subclass InferenceStage to implement the abstract methods
    def _get_inference_worker(self, inf_queue):
        # Intentionally calling the abc empty method for coverage
        super()._get_inference_worker(inf_queue)
        return IW(inf_queue)


def _mk_control_message(mess_count=1, count=1):
    total_message_count = mess_count
    total_tensor_count = count

    df = cudf.DataFrame(list(range(total_message_count)), columns=["col1"])
    msg = ControlMessage()
    msg.payload(MessageMeta(df))
    msg.tensors(
        InferenceMemory(count=total_tensor_count,
                        tensors={
                            "probs": cp.random.rand(total_tensor_count, 2),
                            "seq_ids": cp.tile(cp.expand_dims(cp.arange(0, total_tensor_count), axis=1), (1, 3))
                        }))
    return msg


def test_constructor(config):
    config.feature_length = 128
    config.num_threads = 17
    config.model_max_batch_size = 256

    inf_stage = InferenceStageT(config)
    assert inf_stage._fea_length == 128
    assert inf_stage._thread_count == 17
    assert inf_stage._max_batch_size == 256
    assert inf_stage.name == "inference"

    # Just ensure that we get a valid non-empty tuple
    accepted_types = inf_stage.accepted_types()
    assert isinstance(accepted_types, tuple)
    assert len(accepted_types) > 0

    pytest.raises(NotImplementedError, inf_stage._get_cpp_inference_node, None)


def test_stop(config):
    mock_workers = [mock.MagicMock() for _ in range(5)]
    inf_stage = InferenceStageT(config)
    inf_stage._workers = mock_workers  # pylint: disable=attribute-defined-outside-init

    inf_stage.stop()
    for worker in mock_workers:
        worker.stop.assert_called_once()

    assert inf_stage._inf_queue.is_closed()


def test_join(config):
    mock_workers = [mock.AsyncMock() for _ in range(5)]
    inf_stage = InferenceStageT(config)
    inf_stage._workers = mock_workers  # pylint: disable=attribute-defined-outside-init

    asyncio.run(inf_stage.join())
    for worker in mock_workers:
        worker.join.assert_awaited_once()


@pytest.mark.gpu_mode
def test_convert_one_response():
    # Test ControlMessage
    # Test first branch where `inf.mess_count == inf.count`
    mem = ResponseMemory(count=4, tensors={"probs": cp.zeros((4, 3))})

    inf = _mk_control_message(mess_count=4, count=4)
    res = ResponseMemory(count=4, tensors={"probs": cp.random.rand(4, 3)})
    output = _mk_control_message(mess_count=4, count=4)
    output.tensors(mem)
    batch_offset = 0

    cm = InferenceStageT._convert_one_response(output, inf, res, batch_offset)
    assert cm.payload() == inf.payload()
    assert cm.payload().count == 4
    assert cm.tensor_count() == 4
    assert cp.all(cm.tensors().get_tensor("probs") == res.get_tensor("probs"))

    # Test for the second branch
    inf = _mk_control_message(mess_count=2, count=3)
    inf.tensors().set_tensor("seq_ids", cp.array([[0], [1], [1]]))
    res = ResponseMemory(count=3, tensors={"probs": cp.array([[0, 0.6, 0.7], [5.6, 4.4, 9.2], [4.5, 6.7, 8.9]])})

    mem = ResponseMemory(count=2, tensors={"probs": cp.zeros((2, 3))})
    output = _mk_control_message(mess_count=2, count=3)
    output.tensors(mem)
    cm = InferenceStageT._convert_one_response(output, inf, res, batch_offset)
    assert cm.tensors().get_tensor("probs").tolist() == [[0, 0.6, 0.7], [5.6, 6.7, 9.2]]


def test_convert_one_response_error():
    inf = _mk_control_message(mess_count=2, count=2)
    res = _mk_control_message(mess_count=1, count=1)
    output = inf
    batch_offset = 0

    with pytest.raises(AssertionError):
        InferenceStageT._convert_one_response(output, inf, res.tensors(), batch_offset)
