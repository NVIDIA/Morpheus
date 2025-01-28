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

import pathlib
import typing

import cupy as cp

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.messages.memory.tensor_memory import TensorMemory
from morpheus.stages.inference.inference_stage import InferenceStage
from morpheus.stages.inference.inference_stage import InferenceWorker
from morpheus.utils.producer_consumer_queue import ProducerConsumerQueue

try:
    import torch
    from torch.utils.dlpack import from_dlpack
    from torch.utils.dlpack import to_dlpack
except ImportError:
    print(("PyTorch Not Found! PyTorch must be installed to use the PyTorchInferenceStage. "
           "Due to the limited CUDA options available in the PyTorch stable versions, "
           "it must be manually installed by the user. "
           "Please see the Getting Started Guide: https://pytorch.org/get-started/locally/"))
    raise


class _PyTorchInferenceWorker(InferenceWorker):
    """
    Inference worker used by PyTorchInferenceStage.

    Parameters
    ----------
    inf_queue : `morpheus.utils.producer_consumer_queue.ProducerConsumerQueue`
        Inference queue.
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    model_filename : str
        Model file path.
    """

    def __init__(self, inf_queue: ProducerConsumerQueue, c: Config, model_filename: str):
        super().__init__(inf_queue)

        self._max_batch_size = c.model_max_batch_size
        self._seq_length = c.feature_length
        self._model_filename: str = model_filename

        self._model = None
        self._outputs = {}

        # Use this to cache the output size
        self._output_size = None

    def init(self):

        # Load the model into CUDA memory
        self._model = torch.load(self._model_filename).to('cuda')

    def calc_output_dims(self, msg: ControlMessage) -> typing.Tuple:
        input_ids = msg.tensors().get_tensor("input_ids")
        input_mask = msg.tensors().get_tensor("input_mask")
        count = msg.tensor_count()
        # If we haven't cached the output dimension, do that here
        if (not self._output_size):
            test_intput_ids_shape = (self._max_batch_size, ) + input_ids.shape[1:]
            test_input_mask_shape = (self._max_batch_size, ) + input_mask.shape[1:]

            test_outputs = self._model(torch.randint(65000, (test_intput_ids_shape), dtype=torch.long).cuda(),
                                       token_type_ids=None,
                                       attention_mask=torch.ones(test_input_mask_shape).cuda())

            # Send random input through the model
            self._output_size = test_outputs[0].data.shape

        return (count, self._outputs[list(self._outputs.keys())[0]].shape[1])

    def process(self, batch: ControlMessage, callback: typing.Callable[[TensorMemory], None]):
        input_ids = batch.tensors().get_tensor("input_ids")
        input_mask = batch.tensors().get_tensor("input_mask")
        count = batch.tensor_count()

        # convert from cupy to torch tensor using dlpack
        input_ids = from_dlpack(input_ids.astype(cp.float).toDlpack()).type(torch.long)
        attention_mask = from_dlpack(input_mask.astype(cp.float).toDlpack()).type(torch.long)

        with torch.no_grad():
            logits = self._model(input_ids, token_type_ids=None, attention_mask=attention_mask)[0]
            probs = torch.sigmoid(logits)

        probs_cp = cp.fromDlpack(to_dlpack(probs))

        # Ensure that we are of the shape `[Batch Size, Num Labels]`
        if (len(probs_cp.shape) == 1):
            probs_cp = cp.expand_dims(probs_cp, axis=1)

        response_mem = TensorMemory(count=count, tensors={'probs': probs_cp})

        # Return the response
        callback(response_mem)


@register_stage("inf-pytorch", modes=[PipelineModes.FIL, PipelineModes.NLP, PipelineModes.OTHER])
class PyTorchInferenceStage(InferenceStage):
    """
    Perform inference with PyTorch.

    Pipeline stage for PyTorch inferencing. Uses `PyTorchInference` to process inference inputs using PyTorch.
    Inference outputs are run through sigmoid function to calculate probabilities which are then returned with response.

    Parameters
    ----------
    model_filename : pathlib.Path, exists = True, dir_okay = False
        Model file path.
    """

    def __init__(self, c: Config, model_filename: pathlib.Path):
        super().__init__(c)

        self._config = c
        self._model_filename = str(model_filename)

    def _get_inference_worker(self, inf_queue: ProducerConsumerQueue) -> InferenceWorker:

        return _PyTorchInferenceWorker(inf_queue, self._config, model_filename=self._model_filename)
