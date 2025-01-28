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

import cupy as cp
import pytest

import cudf

from _utils.inference_worker import IW
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.messages import TensorMemory
from morpheus.stages.inference import inference_stage
from morpheus.utils.producer_consumer_queue import ProducerConsumerQueue


def test_constructor():
    queue = ProducerConsumerQueue()
    worker = inference_stage.InferenceWorker(queue)
    assert worker._inf_queue is queue

    # Call empty methods
    worker.init()
    worker.stop()


@pytest.mark.gpu_mode
@pytest.mark.usefixtures("config")
def test_build_output_message():

    # Pylint currently fails to work with classmethod: https://github.com/pylint-dev/pylint/issues/981
    # pylint: disable=no-member

    queue = ProducerConsumerQueue()
    worker = IW(queue)

    num_records = 10
    msg = ControlMessage()
    # input_df = pd.DataFrame(data={'v': (i * 2 for i in range(num_records))})
    df = cudf.DataFrame({
        'v': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18], 'score_test': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    })
    msg.payload(MessageMeta(df))

    input__0 = cp.array([[0.], [2.], [4.], [6.], [8.], [10.], [12.], [14.], [16.], [18.]])
    seq_ids = cp.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0], [6, 0, 0], [7, 0, 0],
                        [8, 0, 0], [9, 0, 0]])
    msg.tensors(TensorMemory(count=num_records, tensors={'input__0': input__0, 'seq_ids': seq_ids}))

    output_message = worker.build_output_message(msg)

    assert (output_message.payload().df.equals(df))
    assert (cp.array_equal(output_message.tensors().get_tensor('probs'), cp.zeros((10, 2))))
