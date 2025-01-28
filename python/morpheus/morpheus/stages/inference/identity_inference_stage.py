# Copyright (c) 2021-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import typing

import cupy as cp

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.messages import TensorMemory
from morpheus.stages.inference.inference_stage import InferenceStage
from morpheus.stages.inference.inference_stage import InferenceWorker
from morpheus.utils.producer_consumer_queue import ProducerConsumerQueue


class _IdentityInferenceWorker(InferenceWorker):
    """
    Worker used by IdentityInferenceStage to set inference probabilities to zeros.

    Parameters
    ----------
    inf_queue : `morpheus.utils.producer_consumer_queueProducerConsumerQueue`
        Inference queue.
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    """

    def __init__(self, inf_queue: ProducerConsumerQueue, c: Config):
        super().__init__(inf_queue)

        self._max_batch_size = c.model_max_batch_size
        self._seq_length = c.feature_length

    def calc_output_dims(self, msg: ControlMessage) -> typing.Tuple:
        return (msg.tensor_count(), self._seq_length)

    def process(self, batch: ControlMessage, callback: typing.Callable[[TensorMemory], None]):

        def tmp(batch: ControlMessage, f):
            count = batch.tensor_count()
            f(TensorMemory(
                count=count,
                tensors={'probs': cp.zeros((count, self._seq_length), dtype=cp.float32)},
            ))

        # Call directly instead of enqueing
        tmp(batch, callback)


@register_stage("inf-identity", modes=[PipelineModes.FIL, PipelineModes.NLP, PipelineModes.OTHER])
class IdentityInferenceStage(InferenceStage):
    """
    Perform inference for testing that performs a no-op.

    Inference stage that simply returns output of zeros with same dimensions as input. Should only be used for testing.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._config = c

    def _get_inference_worker(self, inf_queue: ProducerConsumerQueue) -> InferenceWorker:
        return _IdentityInferenceWorker(inf_queue=inf_queue, c=self._config)
