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

import logging

import cupy as cp
import numpy as np
import tritonclient.grpc as tritonclient
from scipy.special import softmax

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.messages import TensorMemory
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceWorker
from morpheus.utils.producer_consumer_queue import ProducerConsumerQueue

logger = logging.getLogger(__name__)


class TritonInferenceLogParsing(TritonInferenceWorker):
    """
    This class extends TritonInference to deal with scenario-specific NLP models inference requests like building
    response.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    model_name : str
        Name of the model specifies which model can handle the inference requests that are sent to Triton
        inference server.
    server_url : str
        Triton server gRPC URL including the port.
    inout_mapping : typing.Dict[str, str]
        Dictionary used to map pipeline input/output names to Triton input/output names. Use this if the
        Morpheus names do not match the model
    use_shared_memory: bool, default = True
        Whether or not to use CUDA Shared IPC Memory for transferring data to Triton. Using CUDA IPC reduces network
        transfer time but requires that Morpheus and Triton are located on the same machine
    needs_logits : bool, default = True
        Determines whether a logits calculation is needed for the value returned by the Triton inference response.
    """

    def build_output_message(self, msg: ControlMessage) -> ControlMessage:
        seq_ids = cp.zeros((msg.tensor_count(), 3), dtype=cp.uint32)
        seq_ids[:, 0] = cp.arange(0, msg.tensor_count(), dtype=cp.uint32)
        seq_ids[:, 2] = msg.tensors().get_tensor('seq_ids')[:, 2]

        memory = TensorMemory(
            count=msg.tensor_count(),
            tensors={
                'confidences': cp.zeros((msg.tensor_count(), self._inputs[list(self._inputs.keys())[0]].shape[1])),
                'labels': cp.zeros((msg.tensor_count(), self._inputs[list(self._inputs.keys())[0]].shape[1])),
                'input_ids': cp.zeros((msg.tensor_count(), msg.tensors().get_tensor('input_ids').shape[1])),
                'seq_ids': seq_ids
            })

        resp = ControlMessage(msg)
        resp.payload(msg.payload())
        resp.tensors(memory)

        return resp

    def _build_response(self, batch: ControlMessage, result: tritonclient.InferResult) -> TensorMemory:

        outputs = {output.mapped_name: result.as_numpy(output.name) for output in self._outputs.values()}
        outputs = {key: softmax(val, axis=2) for key, val in outputs.items()}
        confidences = {key: np.amax(val, axis=2) for key, val in outputs.items()}
        labels = {key: np.argmax(val, axis=2) for key, val in outputs.items()}

        return TensorMemory(count=outputs[list(outputs.keys())[0]].shape[0],
                            tensors={
                                'confidences': cp.array(confidences[list(outputs.keys())[0]]),
                                'labels': cp.array(labels[list(outputs.keys())[0]])
                            })


@register_stage("inf-logparsing", modes=[PipelineModes.NLP])
class LogParsingInferenceStage(TritonInferenceStage):
    """
    NLP Triton inference stage for log parsing pipeline.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    model_name : str
        Name of the model specifies which model can handle the inference requests that are sent to Triton inference
        server.
    server_url : str
        Triton server URL
    force_convert_inputs : bool, default = False
        Instructs the stage to convert the incoming data to the same format that Triton is expecting. If set to False,
        data will only be converted if it would not result in the loss of data.
    use_shared_memory: bool, default = False, is_flag = True
        Whether or not to use CUDA Shared IPC Memory for transferring data to Triton. Using CUDA IPC reduces network
        transfer time but requires that Morpheus and Triton are located on the same machine
    needs_logits : bool, default = True, is_flag = True
        Determines whether a logits calculation is needed for the value returned by the Triton inference response.
    inout_mapping : dict[str, str], optional
        Dictionary used to map pipeline input/output names to Triton input/output names. Use this if the
        Morpheus names do not match the model.
    """

    def __init__(self,
                 c: Config,
                 model_name: str,
                 server_url: str,
                 force_convert_inputs: bool = False,
                 use_shared_memory: bool = False,
                 needs_logits: bool = True,
                 inout_mapping: dict[str, str] = None):
        super().__init__(c,
                         model_name=model_name,
                         server_url=server_url,
                         force_convert_inputs=force_convert_inputs,
                         use_shared_memory=use_shared_memory,
                         needs_logits=needs_logits,
                         inout_mapping=inout_mapping)

    def supports_cpp_node(self) -> bool:
        return False

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(ControlMessage)

    @staticmethod
    def _convert_one_response(output: ControlMessage, inf: ControlMessage, res: TensorMemory,
                              batch_offset: int) -> ControlMessage:
        memory = output.tensors()

        out_seq_ids = memory.get_tensor('seq_ids')
        input_ids = memory.get_tensor('input_ids')
        confidences = memory.get_tensor('confidences')
        labels = memory.get_tensor('labels')

        seq_ids = inf.tensors().get_tensor('seq_ids')

        seq_offset = seq_ids[0, 0].item()
        seq_count = seq_ids[-1, 0].item() + 1 - seq_offset

        input_ids[batch_offset:inf.tensor_count() + batch_offset, :] = inf.tensors().get_tensor('input_ids')
        out_seq_ids[batch_offset:inf.tensor_count() + batch_offset, :] = seq_ids

        resp_confidences = res.get_tensor('confidences')
        resp_labels = res.get_tensor('labels')

        # Two scenarios:
        if (inf.payload().count == inf.tensor_count()):
            assert seq_count == res.count
            confidences[batch_offset:inf.tensor_count() + batch_offset, :] = resp_confidences
            labels[batch_offset:inf.tensor_count() + batch_offset, :] = resp_labels
        else:
            assert inf.tensor_count() == res.count

            mess_ids = seq_ids[:, 0].get().tolist()

            for i, idx in enumerate(mess_ids):
                confidences[idx, :] = cp.maximum(confidences[idx, :], resp_confidences[i, :])
                labels[idx, :] = cp.maximum(labels[idx, :], resp_labels[i, :])

        resp = ControlMessage(inf)
        resp.payload(inf.payload())
        resp.tensors(memory)
        return resp

    def _get_inference_worker(self, inf_queue: ProducerConsumerQueue) -> TritonInferenceLogParsing:
        return TritonInferenceLogParsing(inf_queue=inf_queue,
                                         c=self._config,
                                         server_url=self._server_url,
                                         model_name=self._model_name,
                                         force_convert_inputs=self._force_convert_inputs,
                                         use_shared_memory=self._use_shared_memory,
                                         input_mapping=self._input_mapping,
                                         output_mapping=self._output_mapping,
                                         needs_logits=self._needs_logits)
