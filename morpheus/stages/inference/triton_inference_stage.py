# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

import base64
import dataclasses
import logging
import queue
import typing
import warnings
from functools import lru_cache
from functools import partial

import cupy as cp
import mrc
import numpy as np
import tritonclient.grpc as tritonclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype

import morpheus._lib.stages as _stages
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.messages.memory.tensor_memory import TensorMemory
from morpheus.stages.inference.inference_stage import InferenceStage
from morpheus.stages.inference.inference_stage import InferenceWorker
from morpheus.utils.producer_consumer_queue import ProducerConsumerQueue

_T = typing.TypeVar("_T")

logger = logging.getLogger(__name__)


@lru_cache(None)
def _notify_dtype_once(model_name: str, input_name: str, triton_dtype: cp.dtype, data_dtype: cp.dtype):
    can_convert = cp.can_cast(data_dtype, triton_dtype, casting="safe")

    msg = "Unexpected dtype for Triton input. "

    if (can_convert):
        msg += "Automatically converting dtype since no data loss will occur. "
    else:
        msg += "Cannot automatically convert dtype due to loss of data. "

    msg += "Model: '%s', Input Name: '%s', Expected dtype: %s, Actual dtype: %s"
    msg_args = (model_name, input_name, str(triton_dtype), str(data_dtype))

    if (can_convert):
        logger.warning(msg, *msg_args)
    else:
        raise RuntimeError(msg % msg_args)


@dataclasses.dataclass()
class TritonInOut:
    """
    Data class for model input and output configuration.

    Parameters
    ----------
    name : str
        Name of the input/output in the model.
    bytes : int
        Total bytes.
    datatype : str
        Triton string for datatype.
    shape : typing.List[int]
        Shape of input/output.
    mapped_name : str
        Name of the input/output in the pipeline.
    offset : int
        Offset, default value is 0.
    ptr : cp.cuda.MemoryPointer
        Cupy cuda memory pointer for the input/output.

    """
    name: str  # Name of the input/output in the model
    bytes: int  # Total bytes
    datatype: str  # Triton string for datatype
    shape: typing.List[int]
    mapped_name: str  # Name of the input/output in the pipeline
    offset: int = 0
    ptr: cp.cuda.MemoryPointer = None


class ResourcePool(typing.Generic[_T]):
    """
    This class provides a bounded pool of resources. Users of the pool can borrow a resource where they will
    get exclusive access to that resource until it is returned. New objects will be created if the pool is
    empty when a user requets to borrow a resource. If the max size has been hit, the user thread will be
    blocked until another thread returns a resource.

    Parameters
    ----------
    create_fn : typing.Callable[[], typing.Any]

        Function used to create new resource objects when needed.

    max_size : int, default = 10000

        Maximum number of messages in a queue.

    """

    def __init__(self, create_fn: typing.Callable[[], _T], max_size: int = 1000):
        self._create_fn = create_fn
        self._max_size = max_size
        self._added_count = 0

        self._queue: ProducerConsumerQueue[_T] = ProducerConsumerQueue(maxsize=self._max_size)

    def _add_item(self):
        try:
            # Hold the queue mutex while we create this
            with self._queue.mutex:

                # Only add it if we have room. Otherwise we allocate memory each time we try to exceed the size
                if (self._added_count < self._max_size):
                    self._queue.put_nowait(self._create_fn())
                    self._added_count += 1

        except queue.Full:
            logger.error(
                "Failed to add item to the Triton ResourcePool. The ResourcePool and queue size are out of sync.")
            raise

    @property
    def added_count(self):
        """
        The number of items that have been generated by the pool. Starts at 0 and increases for ever borrow request when
        the current pool is empty.

        Returns
        -------
        int
            Current number of added items.
        """
        return self._added_count

    def borrow_obj(self, timeout: float = None) -> _T:
        """
        Returns an item from the pool. If the pool is empty, a new item will be created and returned.

        Returns
        -------
        obj
            Item from the queue.
        """

        try:
            return self._queue.get_nowait()
        except queue.Empty:
            # Now try and create one
            self._add_item()

            return self._queue.get(timeout=timeout)

    def return_obj(self, obj: _T):
        """
        Returns a borrowed item back to the pool to be used by new calls to `borrow()`.

        Parameters
        ----------
        obj
            An item to be added to the queue.
        """

        # Use put_nowait here because we should never exceed the size and this should fail instead of blocking
        self._queue.put_nowait(obj)


class InputWrapper:
    """
    This class is a wrapper around a CUDA shared memory object shared between this process and a Triton server instance.
    Since the Triton server only accepts numpy arrays as inputs, we can use this special class to pass memory references
    of inputs on the device to the server without having to go to the host eliminating serialization and network
    overhead.

    Parameters
    ----------
    client : tritonclient.InferenceServerClient
        Triton inference server client instance.
    model_name : str
        Name of the model. Specifies which model can handle the inference requests that are sent to Triton
        inference server.
    config : typing.Dict[str, `TritonInOut`]
        Model input and output configuration. Keys represent the input/output names. Values will be a
        `TritonInOut` object.

    """

    def __init__(
            self,
            client: tritonclient.InferenceServerClient,  # pylint: disable=unused-argument
            model_name: str,
            config: typing.Dict[str, TritonInOut]):
        self._config = config.copy()

        self._total_bytes = 0

        for key in self._config.keys():
            self._config[key].offset = self._total_bytes
            self._total_bytes += self._config[key].bytes

        self.model_name = model_name

    def get_bytes(self, name: str):
        """
        Get the bytes needed for a particular input/output.

        Parameters
        ----------
        name : str
            Configuration name.

        Returns
        -------
        bytes
            Configuration as bytes.

        """
        return self._config[name].bytes

    def get_offset(self, name: str):
        """
        Get the offset needed for a particular input/output.

        Parameters
        ----------
        name : str
            Configuration input/output name.

        Returns
        -------
        int
            Configuration offset.

        """
        return self._config[name].offset

    def get_ptr(self, name: str) -> cp.cuda.MemoryPointer:
        """
        Returns the `cupy.cuda.MemoryPointer` object to the internal `ShmWrapper` for the specified
        input/output name.

        :meta public:

        Parameters
        ----------
            name : str
                Input/output name.

        Returns
        -------
            cp.cuda.MemoryPointer :
                Returns the shared memory pointer for this input/output.

        """
        return self._config[name].ptr

    def _convert_data(self, name: str, data: cp.ndarray, force_convert_inputs: bool):
        """
        This helper function builds a Triton InferInput object that can be directly used by `tritonclient.async_infer`.
        Utilizes the config option passed in the constructor to determine the shape/size/type.

        Parameters
        ----------
        name : str
            Inference input name.
        data : cupy.ndarray
            Inference input data.
        force_convert_inputs: bool
            Whether or not to convert the inputs to the type specified by Triton. This will happen automatically if no
            data would be lost in the conversion (i.e., float -> double). Set this to True to convert the input even if
            data would be lost (i.e., double -> float).

        """

        expected_dtype = cp.dtype(triton_to_np_dtype(self._config[name].datatype))

        if (expected_dtype != data.dtype):

            # See if we can auto convert without loss if force_convert_inputs is False
            if (not force_convert_inputs):
                _notify_dtype_once(self.model_name, name, expected_dtype, data.dtype)

            data = data.astype(expected_dtype)

        return data

    def build_input(self, name: str, data: cp.ndarray, force_convert_inputs: bool) -> tritonclient.InferInput:
        """
        This helper function builds a Triton InferInput object that can be directly used by `tritonclient.async_infer`.
        Utilizes the config option passed in the constructor to determine the shape/size/type.

        Parameters
        ----------
        name : str
            Inference input name.
        data : cp.ndarray
            Inference input data.
        force_convert_inputs: bool
            Whether or not to convert the inputs to the type specified by Triton. This will happen automatically if no
            data would be lost in the conversion (i.e., float -> double). Set this to True to convert the input even if
            data would be lost (i.e., double -> float).

        """

        triton_input = tritonclient.InferInput(name, list(data.shape), self._config[name].datatype)

        data = self._convert_data(name, data, force_convert_inputs)

        # Set the memory using numpy
        triton_input.set_data_from_numpy(data.get())

        return triton_input


class ShmInputWrapper(InputWrapper):
    """
    This class is a wrapper around a CUDA shared memory object shared between this process and a Triton server instance.
    Since the Triton server only accepts numpy arrays as inputs, we can use this special class to pass memory references
    of inputs on the device to the server without having to go to the host eliminating serialization and network
    overhead.

    Parameters
    ----------
    client : tritonclient.InferenceServerClient
        Triton inference server client instance.
    model_name : str
        Name of the model. Specifies which model can handle the inference requests that are sent to Triton
        inference server.
    config : typing.Dict[str, `TritonInOut`]
        Model input and output configuration. Keys represent the input/output names. Values will be a
        `TritonInOut` object.

    """
    total_count = 0

    def __init__(self,
                 client: tritonclient.InferenceServerClient,
                 model_name: str,
                 config: typing.Dict[str, TritonInOut]):
        super().__init__(client, model_name, config)

        # Now create the necessary shared memory bits
        self.region_name = f"{model_name}_{ShmInputWrapper.total_count}"
        ShmInputWrapper.total_count += 1

        # Allocate the total memory
        self._memory: cp.cuda.Memory = cp.cuda.alloc(self._total_bytes).mem

        # Get memory pointers for each object
        for key in self._config.keys():
            self._config[key].ptr = cp.cuda.MemoryPointer(self._memory, self._config[key].offset)

        # Now get the registered IPC handle
        self._ipc_handle = cp.cuda.runtime.ipcGetMemHandle(self._memory.ptr)  # pylint: disable=c-extension-no-member

        # Finally, regester this memory with the server. Must be base64 for some reason???
        client.register_cuda_shared_memory(self.region_name, base64.b64encode(self._ipc_handle), 0, self._total_bytes)

    def build_input(self, name: str, data: cp.ndarray, force_convert_inputs: bool) -> tritonclient.InferInput:
        """
        This helper function builds a Triton InferInput object that can be directly used by `tritonclient.async_infer`.
        Utilizes the config option passed in the constructor to determine the shape/size/type.

        Parameters
        ----------
        name : str
            Inference input name.
        data : cupy.ndarray
            Inference input data.
        force_convert_inputs: bool
            Whether or not to convert the inputs to the type specified by Triton. This will happen automatically if no
            data would be lost in the conversion (i.e., float -> double). Set this to True to convert the input even if
            data would be lost (i.e., double -> float).

        """

        triton_input = tritonclient.InferInput(name, list(data.shape), self._config[name].datatype)

        data = self._convert_data(name, data, force_convert_inputs)

        # Set the data
        self.get_ptr(name).copy_from_device(data.data, data.nbytes)

        # Configure the shared memory
        triton_input.set_shared_memory(self.region_name, data.nbytes, self.get_offset(name))

        return triton_input


# This class is exclusively run in the worker thread. Separating the classes helps keeps the threads separate
class TritonInferenceWorker(InferenceWorker):
    """
    Inference worker class for all Triton inference server requests.

    Parameters
    ----------
    inf_queue : `morpheus.utils.producer_consumer_queue.ProducerConsumerQueue`
        Inference queue.
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    model_name : str
        Name of the model specifies which model can handle the inference requests that are sent to Triton
        inference server.
    server_url : str
        Triton server gRPC URL including the port.
    force_convert_inputs: bool
        Whether to convert the inputs to the type specified by Triton. This will happen automatically if no
        data would be lost in the conversion (i.e., float -> double). Set this to True to convert the input even if
        data would be lost (i.e., double -> float).
    inout_mapping : dict[str, str]
        Dictionary used to map pipeline input/output names to Triton input/output names. Use this if the
        Morpheus names do not match the model.
    use_shared_memory: bool, default = False
        Whether to use CUDA Shared IPC Memory for transferring data to Triton. Using CUDA IPC reduces network
        transfer time but requires that Morpheus and Triton are located on the same machine.
    needs_logits : bool, default = False
        Determines whether a logits calculation is needed for the value returned by the Triton inference response.
    """

    def __init__(self,
                 inf_queue: ProducerConsumerQueue,
                 c: Config,
                 model_name: str,
                 server_url: str,
                 force_convert_inputs: bool,
                 input_mapping: dict[str, str] = None,
                 output_mapping: dict[str, str] = None,
                 use_shared_memory: bool = False,
                 needs_logits: bool = False):
        super().__init__(inf_queue)

        self._model_name = model_name
        self._server_url = server_url
        self._input_mapping = input_mapping or {}
        self._output_mapping = output_mapping or {}
        self._use_shared_memory = use_shared_memory

        self._max_batch_size = c.model_max_batch_size
        self._fea_length = c.feature_length
        self._force_convert_inputs = force_convert_inputs

        # Whether the returned value needs a logits calc for the response
        self._needs_logits = needs_logits

        self._inputs: typing.Dict[str, TritonInOut] = {}
        self._outputs: typing.Dict[str, TritonInOut] = {}

        self._triton_client: tritonclient.InferenceServerClient = None
        self._mem_pool: ResourcePool = None

    @classmethod
    def supports_cpp_node(cls):
        # Enable support by default
        return True

    @property
    def needs_logits(self) -> bool:
        return self._needs_logits

    def init(self):
        """
        This function instantiate triton client and memory allocation for inference input and output.
        """

        self._triton_client = tritonclient.InferenceServerClient(url=self._server_url, verbose=False)

        try:
            assert self._triton_client.is_server_live() and self._triton_client.is_server_ready(), \
                "Server is not in ready state"

            assert self._triton_client.is_model_ready(self._model_name), \
                f"Triton model {self._model_name} is not ready"

            # To make sure no shared memory regions are registered with the server.
            self._triton_client.unregister_system_shared_memory()
            self._triton_client.unregister_cuda_shared_memory()

            model_meta = self._triton_client.get_model_metadata(self._model_name, as_json=True)
            model_config = self._triton_client.get_model_config(self._model_name, as_json=True)["config"]

            # Make sure the inputs/outputs match our config
            if (int(model_meta["inputs"][0]["shape"][-1]) != self._fea_length):
                raise RuntimeError(f"Mismatched Sequence Length. Config specified {self._fea_length} but model"
                                   f" specified {int(model_meta['inputs'][0]['shape'][-1])}")

            # Check batch size
            if (model_config.get("max_batch_size", 0) != self._max_batch_size):

                # If the model is more, that's fine. Gen warning
                if (model_config["max_batch_size"] > self._max_batch_size):
                    warnings.warn(
                        f"Model max batch size ({model_config['max_batch_size']}) is more than configured max batch "
                        f"size ({self._max_batch_size}). May result in sub optimal performance")

                # If the model is less, raise error. Cant send more to Triton than the max batch size
                if (model_config["max_batch_size"] < self._max_batch_size):
                    raise RuntimeError(
                        f"Model max batch size ({model_config['max_batch_size']}) is less than configured max batch"
                        f" size ({self._max_batch_size}). Reduce max batch size to be less than or equal to model max"
                        " batch size.")

            shm_config = {}

            def build_inout(x: dict, mapping: dict[str, str]):
                num_bytes = np.dtype(triton_to_np_dtype(x["datatype"])).itemsize

                shape = []

                for y in x["shape"]:
                    y_int = int(y)

                    if (y_int == -1):
                        y_int = self._max_batch_size

                    shape.append(y_int)

                    num_bytes *= y_int

                mapped_name = x["name"] if x["name"] not in mapping else mapping[x["name"]]

                return TritonInOut(name=x["name"],
                                   bytes=num_bytes,
                                   datatype=x["datatype"],
                                   shape=shape,
                                   mapped_name=mapped_name)

            for x in model_meta["inputs"]:
                self._inputs[x["name"]] = build_inout(x, self._input_mapping)

            for x in model_meta["outputs"]:
                assert x["name"] not in self._inputs, "Input/Output names must be unique from eachother"

                self._outputs[x["name"]] = build_inout(x, self._output_mapping)

            # Combine the inputs/outputs for the shared memory
            shm_config = {**self._inputs, **self._outputs}

            if (self._use_shared_memory):

                def create_wrapper():
                    return ShmInputWrapper(self._triton_client, self._model_name, shm_config)
            else:

                def create_wrapper():
                    return InputWrapper(self._triton_client, self._model_name, shm_config)

            self._mem_pool = ResourcePool(create_fn=create_wrapper, max_size=1000)

        except InferenceServerException as ex:
            logger.exception("Exception occurred while coordinating with Triton. Exception message: \n%s\n",
                             ex,
                             exc_info=ex)
            raise ex

    def calc_output_dims(self, msg: ControlMessage) -> typing.Tuple:
        return (msg.tensors().count, self._outputs[list(self._outputs.keys())[0]].shape[1])

    def _build_response(
            self,
            batch: ControlMessage,  # pylint: disable=unused-argument
            result: tritonclient.InferResult) -> TensorMemory:
        output = {output.mapped_name: result.as_numpy(output.name) for output in self._outputs.values()}

        # Make sure we have at least 2 dims
        for key, val in output.items():
            if (len(val.shape) == 1):
                output[key] = np.expand_dims(val, 1)

        if (self._needs_logits):
            output = {key: 1.0 / (1.0 + np.exp(-val)) for key, val in output.items()}

        return TensorMemory(
            count=output["probs"].shape[0],
            tensors={'probs': cp.array(output["probs"])}  # For now, only support one output
        )

    # pylint: disable=invalid-name
    def _infer_callback(self,
                        cb: typing.Callable[[TensorMemory], None],
                        m: InputWrapper,
                        b: ControlMessage,
                        result: tritonclient.InferResult,
                        error: tritonclient.InferenceServerException):

        # If its an error, return that here
        if (error is not None):
            raise error

        # Build response
        response_mem = self._build_response(b, result)

        # Call the callback with the memory
        cb(response_mem)

        self._mem_pool.return_obj(m)

    # pylint: enable=invalid-name

    def process(self, batch: ControlMessage, callback: typing.Callable[[TensorMemory], None]):
        """
        This function sends batch of events as a requests to Triton inference server using triton client API.

        Parameters
        ----------
        batch : `morpheus.messages.ControlMessage`
            Mini-batch of inference messages.
        callback : typing.Callable[[`morpheus.pipeline.messages.TensorMemory`], None]
            Callback to set the values for the inference response.

        """
        mem: InputWrapper = self._mem_pool.borrow_obj()

        inputs: typing.List[tritonclient.InferInput] = [
            mem.build_input(input.name,
                            batch.tensors().get_tensor(input.mapped_name),
                            force_convert_inputs=self._force_convert_inputs) for input in self._inputs.values()
        ]

        outputs = [tritonclient.InferRequestedOutput(output.name) for output in self._outputs.values()]

        # Inference call
        self._triton_client.async_infer(model_name=self._model_name,
                                        inputs=inputs,
                                        callback=partial(self._infer_callback, callback, mem, batch),
                                        outputs=outputs)


@register_stage("inf-triton", modes=[PipelineModes.NLP, PipelineModes.FIL, PipelineModes.OTHER])
class TritonInferenceStage(InferenceStage):
    """
    Perform inference with Triton Inference Server.

    This class specifies which inference implementation category (Ex: NLP/FIL) is needed for inferencing.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    model_name : str
        Name of the model specifies which model can handle the inference requests that are sent to Triton inference
        server.
    server_url : str
        Triton server URL.
    force_convert_inputs : bool, default = False
        Instructs the stage to convert the incoming data to the same format that Triton is expecting. If set to False,
        data will only be converted if it would not result in the loss of data.
    use_shared_memory : bool, default = False, is_flag = True
        Whether or not to use CUDA Shared IPC Memory for transferring data to Triton. Using CUDA IPC reduces network
        transfer time but requires that Morpheus and Triton are located on the same machine.
    needs_logits : bool, optional
        Determines whether a logits calculation is needed for the value returned by the Triton inference response. If
        undefined, the value will be inferred based on the pipeline mode, defaulting to `True` for NLP and `False` for
        other modes.
    inout_mapping : dict[str, str], optional
        Dictionary used to map pipeline input/output names to Triton input/output names.
        Use this if the Morpheus names do not match the model.
        If undefined, a default mapping will be used based on the pipeline mode as follows:

        * `FIL`: `{"output__0": "probs"}`

        * `NLP`: `{"attention_mask": "input_mask", "output": "probs"}`

        * All other modes: `{}`

        From the command line this can be specified multiple times for each key/value pair, for example:

            --inout-mapping mask input_mask --inout-mapping output probs

        which will be inroduced as:

            inout_mapping={"mask": "input_mask", "output": "probs"}
    """

    _INFERENCE_WORKER_DEFAULT_INOUT_MAPPING = {
        PipelineModes.FIL: {
            "outputs": {
                "output__0": "probs",
            }
        },
        PipelineModes.NLP: {
            "inputs": {
                "attention_mask": "input_mask",
            }, "outputs": {
                "output": "probs",
            }
        }
    }

    def __init__(self,
                 c: Config,
                 model_name: str,
                 server_url: str,
                 force_convert_inputs: bool = False,
                 use_shared_memory: bool = False,
                 needs_logits: bool = None,
                 inout_mapping: dict[str, str] = None,
                 input_mapping: dict[str, str] = None,
                 output_mapping: dict[str, str] = None):
        super().__init__(c)

        self._config = c

        if needs_logits is None:
            needs_logits = c.mode == PipelineModes.NLP

        input_mapping_ = self._INFERENCE_WORKER_DEFAULT_INOUT_MAPPING.get(c.mode, {}).get("inputs", {})
        output_mapping_ = self._INFERENCE_WORKER_DEFAULT_INOUT_MAPPING.get(c.mode, {}).get("outputs", {})

        if inout_mapping:

            if input_mapping:
                raise RuntimeError(
                    "TritonInferenceStages' `inout_mapping` and `input_mapping` arguments cannot be used together`")

            if output_mapping:
                raise RuntimeError(
                    "TritonInferenceStages' `inout_mapping` and `output_mapping` arguments cannot be used together`")

            warnings.warn(("TritonInferenceStage's `inout_mapping` argument has been deprecated. "
                           "Please use `input_mapping` and/or `output_mapping` instead"),
                          DeprecationWarning)

            input_mapping_.update(inout_mapping)
            output_mapping_.update(inout_mapping)

        if input_mapping is not None:
            input_mapping_.update(input_mapping)

        if output_mapping is not None:
            output_mapping_.update(output_mapping)

        self._server_url = server_url
        self._model_name = model_name
        self._force_convert_inputs = force_convert_inputs
        self._use_shared_memory = use_shared_memory
        self._input_mapping = input_mapping_
        self._output_mapping = output_mapping_
        self._needs_logits = needs_logits

    def supports_cpp_node(self) -> bool:
        # Get the value from the worker class
        if TritonInferenceWorker.supports_cpp_node():
            if not self._use_shared_memory:
                return True

            logger.warning("The C++ implementation of TritonInferenceStage does not support the use_shared_memory "
                           "option. Falling back to Python implementation.")

        return False

    def _get_inference_worker(self, inf_queue: ProducerConsumerQueue) -> TritonInferenceWorker:
        """
        Returns the worker for this stage. Authors of custom sub-classes can override this method to provide a custom
        worker.
        """

        return TritonInferenceWorker(inf_queue=inf_queue,
                                     c=self._config,
                                     server_url=self._server_url,
                                     model_name=self._model_name,
                                     force_convert_inputs=self._force_convert_inputs,
                                     use_shared_memory=self._use_shared_memory,
                                     input_mapping=self._input_mapping,
                                     output_mapping=self._output_mapping,
                                     needs_logits=self._needs_logits)

    def _get_cpp_inference_node(self, builder: mrc.Builder) -> mrc.SegmentObject:
        return _stages.InferenceClientStage(builder,
                                            self.unique_name,
                                            self._server_url,
                                            self._model_name,
                                            self._needs_logits,
                                            self._force_convert_inputs,
                                            self._input_mapping,
                                            self._output_mapping)

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = super()._build_single(builder, input_node)

        return node
