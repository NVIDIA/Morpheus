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
import typing
from functools import partial

import cupy as cp
import mrc
from mrc.core import operators as ops

from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages.memory.tensor_memory import TensorMemory
from morpheus.pipeline.control_message_stage import ControlMessageStage
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.utils.producer_consumer_queue import ProducerConsumerQueue

logger = logging.getLogger(__name__)


class InferenceWorker:
    """
    Base class for providing implementation details for an inference stage. Create inference worker by
    subclassing this and implementing the required abstract methods. Inference stage class can then be
    assigned this worker by implementing _get_inference_worker to return your subclass.

    Parameters
    ----------
    inf_queue : `morpheus.utils.producer_consumer_queue.ProducerConsumerQueue`
        Inference queue.

    """

    def __init__(self, inf_queue: ProducerConsumerQueue) -> None:
        self._inf_queue = inf_queue

    def init(self):
        """
        By overriding this function, the resources necessary for the inference can be initiated.
        Each inference worker calls this function once.
        """

        # Nothing required in base
        pass

    def stop(self):
        """
        Override this function to stop the inference workers or carry out any additional cleanups.
        """

        pass

    def build_output_message(self, msg: ControlMessage) -> ControlMessage:
        """
        Create initial inference response message with result values initialized to zero. Results will be
        set in message as each inference mini-batch is processed.

        Parameters
        ----------
        msg : `morpheus.messages.ControlMessage`
            Batch of ControlMessage.

        Returns
        -------
        `morpheus.messages.ControlMessage`
            Response message with probabilities calculated from inference results.
        """
        dims = self.calc_output_dims(msg)
        output_dims = (msg.payload().count, *dims[1:])

        memory = TensorMemory(count=output_dims[0], tensors={'probs': cp.zeros(output_dims)})
        output_message = ControlMessage(msg)
        output_message.payload(msg.payload())
        output_message.tensors(memory)

        return output_message

    def calc_output_dims(self, msg: ControlMessage) -> tuple:
        """
        Calculates the dimensions of the inference output message data given an input message.

        Parameters
        ----------
        msg : `morpheus.messages.ControlMessage`
            Pipeline inference input batch before splitting into smaller inference batches.

        Returns
        -------
        tuple
            Output dimensions of response.
        """
        raise NotImplementedError("No Python implementation provided by this stage")

    def process(self, batch: ControlMessage, callback: typing.Callable[[TensorMemory], None]):
        """
        Main inference processing function. This function will be called once for each mini-batch. Once the inference is
        complete, the `cb` parameter should be used to set the response value. The callback can be called
        asynchronously.

        Parameters
        ----------
        batch : `morpheus.messages.ControlMessage`
            Mini-batch of inference messages.
        callback : typing.Callable[[`morpheus.pipeline.messages.TensorMemory`], None]
            Callback to set the values for the inference response.

        """
        raise NotImplementedError("No Python implementation provided by this stage")


class InferenceStage(ControlMessageStage):
    """
    This class serves as the base for any inference stage. Inference stages operate differently than other
    stages due to the fact that they operate in a separate thread and have their own batch size which is
    separate from the pipeline batch size. Processing the inference work in a separate thread is necessary to
    support inference types that may require exclusive use of a single thread (i.e., TensorRT) without blocking
    the main thread.

    Changing batch sizes for the inference stage requires breaking messages into smaller slices, running
    inference on the smaller slices, then recombining the inference output into the original batch size. This
    inference base class handles breaking and recombining batches and queing the inference work to be
    processed on another thread.

    Inference stages that derive from this class need to implement the `_get_inference_worker` method which
    returns an implementation of the `InferenceWorker` class. Your `InferenceWorker` class must implement the
    `process` and `calc_output_dims` methods. The `process` methods is where you provide implementation details
    on how to perform inference with the `ControlMessage` batch. The worker uses the `calc_output_dims` to
    calculate the output dimensions of the pipeline batch that inference batch results are appended to.

    To add a C++ implementation for processing inference requests, you must implement the `_get_cpp_inference_node`
    method and implement `supports_cpp_node` in your worker to return True. Your pipeline can then use your C++
    implementation by setting `use_cpp` to True in your pipeline configuration. See developer documentation for
    more details.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    thread_count : int, optional
        Number of threads to use for inference. If not provided, the `num_threads` attribute of the `Config` object
        will be used.
    """

    def __init__(self, c: Config, thread_count: int = None):
        super().__init__(c)

        # GPU only stage, assuming all messages are cuDF/CuPy based
        import cudf
        self._cudf = cudf

        self._fea_length = c.feature_length

        self._thread_count = thread_count or c.num_threads
        self._workers: typing.List[InferenceWorker] = []
        self._inf_queue = ProducerConsumerQueue()

        self._max_batch_size = c.model_max_batch_size

        # Mark these stages to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        return "inference"

    def accepted_types(self) -> tuple:
        """
        Accepted input types to this stage.

        Returns
        -------
        tuple
            Tuple of input types.
        """
        return (ControlMessage, )

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(ControlMessage)

    def supports_cpp_node(self) -> bool:
        # Default to False unless derived classes override this value
        return False

    def _get_inference_worker(self, inf_queue: ProducerConsumerQueue) -> InferenceWorker:
        """
        Returns the main inference worker which manages requests possibly in another thread depending on which mode the
        pipeline is currently operating in.

        :meta public:

        Parameters
        ----------
        inf_queue : `morpheus.utils.producer_consumer_queue.ProducerConsumerQueue`
            Inference request queue.

        Returns
        -------
        `InferenceWorker`
            Inference worker implementation for stage.
        """
        raise NotImplementedError("No Python implementation provided by this stage")

    def _get_cpp_inference_node(self, builder: mrc.Builder) -> mrc.SegmentObject:
        raise NotImplementedError("No C++ node is available for this inference type")

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:

        def py_inference_fn(obs: mrc.Observable, sub: mrc.Subscriber):

            worker = self._get_inference_worker(self._inf_queue)

            worker.init()

            outstanding_requests = 0

            def on_next(message: ControlMessage):
                nonlocal outstanding_requests

                batches = self._split_batches(message, self._max_batch_size)
                output_message = worker.build_output_message(message)

                fut_list = []

                batch_offset = 0

                for batch in batches:
                    outstanding_requests += 1

                    completion_future = mrc.Future()

                    def set_output_fut(resp: TensorMemory, inner_batch, batch_future: mrc.Future):
                        nonlocal outstanding_requests
                        nonlocal batch_offset
                        mess = self._convert_one_response(output_message, inner_batch, resp, batch_offset)
                        batch_offset += inner_batch.tensor_count()
                        outstanding_requests -= 1

                        batch_future.set_result(mess)

                    fut_list.append(completion_future)

                    worker.process(batch, partial(set_output_fut, inner_batch=batch, batch_future=completion_future))

                for f in fut_list:
                    f.result()

                return output_message

            obs.pipe(ops.map(on_next)).subscribe(sub)

            assert outstanding_requests == 0, "Not all inference requests were completed"

        if (self._build_cpp_node()):
            node = self._get_cpp_inference_node(builder)
        else:
            node = builder.make_node(self.unique_name, ops.build(py_inference_fn))

        # Set the concurrency level to be up with the thread count
        node.launch_options.pe_count = self._thread_count
        builder.make_edge(input_node, node)

        return node

    def stop(self):
        """
        Stops the inference workers and closes the inference queue.
        """

        for worker in self._workers:
            worker.stop()

        # Now stop the _inf_queue to unblock workers
        self._inf_queue.close()

    async def join(self):
        """
        On all inference worker threads, this function applies join.
        """

        # Wait for queue to be finished. This does block but it should be fine for now
        self._inf_queue.join()

        # Join all workers
        for worker in self._workers:
            await worker.join()

        return await super().join()

    @staticmethod
    def _split_batches(msg: ControlMessage, max_batch_size: int) -> typing.List[ControlMessage]:
        out_batches = []

        id_array = cp.concatenate([cp.array([-1]), msg.tensors().get_tensor("seq_ids")[:, 0], cp.array([-1])])

        diff_ids = cp.where(id_array[1:] != id_array[:-1])[0]

        diff_ids = diff_ids.tolist()

        head = 0
        tail = 0

        for i in range(1, len(diff_ids)):

            poss_count = diff_ids[i] - diff_ids[head]

            if (poss_count > max_batch_size):
                out_batches.append((diff_ids[head], diff_ids[tail]))

                head = tail

            tail = i

        out_batches.append((diff_ids[head], diff_ids[tail]))

        out_resp = []

        for start, stop in out_batches:
            out_msg = ControlMessage(msg)

            out_msg.payload(msg.payload().get_slice(start, stop))

            out_msg_tensors = TensorMemory(count=stop - start, tensors={})
            for (name, tensor) in msg.tensors().get_tensors().items():
                out_msg_tensors.set_tensor(name, tensor[start:stop])
            out_msg.tensors(out_msg_tensors)

            out_resp.append(out_msg)

        assert len(out_resp) > 0

        return out_resp

    @staticmethod
    def _convert_one_response(output: ControlMessage, inf: ControlMessage, res: TensorMemory,
                              batch_offset: int) -> ControlMessage:  # pylint:disable=unused-argument
        # Make sure we have a continuous list
        # assert inf.mess_offset == saved_offset + saved_count

        memory = output.tensors()

        probs = memory.get_tensor("probs")
        resp_probs = res.get_tensor("probs")

        seq_ids = inf.tensors().get_tensor("seq_ids")

        seq_offset = seq_ids[0, 0].item()
        seq_count = seq_ids[-1, 0].item() + 1 - seq_offset

        # Two scenarios:
        if (inf.payload().count == inf.tensor_count()):
            assert seq_count == res.count

            # In message and out message have same count. Just use probs as is
            probs[seq_offset:seq_offset + seq_count, :] = resp_probs
        else:
            assert inf.tensor_count() == res.count

            mess_ids = seq_ids[:, 0].get().tolist()

            # Out message has more reponses, so we have to do key based blending of probs
            for i, idx in enumerate(mess_ids):
                probs[idx, :] = cp.maximum(probs[idx, :], resp_probs[i, :])

        msg = ControlMessage(inf)
        msg.payload(inf.payload())
        msg.tensors(memory)

        return msg
