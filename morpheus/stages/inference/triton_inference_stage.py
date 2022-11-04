# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
import threading
import typing
import warnings
from abc import abstractmethod
from functools import lru_cache
from functools import partial

import cupy as cp
import numpy as np
import srf
import tritonclient.grpc as tritonclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype

import morpheus._lib.stages as _stages
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import MultiInferenceMessage
from morpheus.messages import ResponseMemory
from morpheus.messages import ResponseMemoryProbs
from morpheus.stages.inference.inference_stage import InferenceStage
from morpheus.stages.inference.inference_stage import InferenceWorker
from morpheus.utils.producer_consumer_queue import ProducerConsumerQueue

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


class ResourcePool:
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

    def __init__(self, create_fn: typing.Callable[[], typing.Any], max_size: int = 1000):
        self._create_fn = create_fn
        self._max_size = max_size
        self._added_count = 0

        self._queue = ProducerConsumerQueue()

        self._adding_condition = threading.Condition(self._queue.mutex)

        self._outstanding = []

    def _borrow(self):
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            # Now try and create one
            with self._queue.mutex:

                # Only add it if we have room
                if (self._added_count < self._max_size):
                    self._queue.put(self._create_fn())
                    self._added_count += 1

            return self._queue.get()

    def borrow(self):
        obj = self._borrow()

        return obj

    def return_obj(self, obj):
        self._queue.put(obj)


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

    def __init__(self,
                 client: tritonclient.InferenceServerClient,
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

    def build_input_http(self, name: str, data: cp.ndarray, force_convert_inputs: bool) -> tritonclient_http.InferInput:
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

        triton_input = tritonclient_http.InferInput(name, list(data.shape), self._config[name].datatype)

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
        self.region_name = model_name + "_{}".format(ShmInputWrapper.total_count)
        ShmInputWrapper.total_count += 1

        # Allocate the total memory
        self._memory: cp.cuda.Memory = cp.cuda.alloc(self._total_bytes).mem

        # Get memory pointers for each object
        for key in self._config.keys():
            self._config[key].ptr = cp.cuda.MemoryPointer(self._memory, self._config[key].offset)

        # Now get the registered IPC handle
        self._ipc_handle = cp.cuda.runtime.ipcGetMemHandle(self._memory.ptr)

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
class _TritonInferenceWorker(InferenceWorker):
    """
    This is a base class for all Triton inference server requests.

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
        Whether or not to convert the inputs to the type specified by Triton. This will happen automatically if no
        data would be lost in the conversion (i.e., float -> double). Set this to True to convert the input even if
        data would be lost (i.e., double -> float).
    inout_mapping : typing.Dict[str, str]
        Dictionary used to map pipeline input/output names to Triton input/output names. Use this if the
        Morpheus names do not match the model.
    use_shared_memory: bool, default = False
        Whether or not to use CUDA Shared IPC Memory for transferring data to Triton. Using CUDA IPC reduces network
        transfer time but requires that Morpheus and Triton are located on the same machine.
    response_log_filename: str, default = None
        Saves the responses from Triton into a CSV file to support debugging and mocking. Use `None` to disable this
        feature. Requires the stage to be run in python mode.
    """

    __REQUEST_COUNTER = AtomicInteger()

    def __init__(self,
                 inf_queue: ProducerConsumerQueue,
                 c: Config,
                 model_name: str,
                 server_url: str,
                 force_convert_inputs: bool,
                 inout_mapping: typing.Dict[str, str] = None,
                 use_shared_memory: bool = False,
                 response_log_filename: str = None):
        super().__init__(inf_queue)

        # Combine the class defaults with any user supplied ones
        default_mapping = type(self).default_inout_mapping()

        default_mapping.update(inout_mapping if inout_mapping is not None else {})

        self._model_name = model_name
        self._server_url = server_url
        self._inout_mapping = default_mapping
        self._use_shared_memory = use_shared_memory

        self._requires_seg_ids = False

        self._max_batch_size = c.model_max_batch_size
        self._fea_length = c.feature_length
        self._force_convert_inputs = force_convert_inputs

        # Whether or not the returned value needs a logits calc for the response
        self._needs_logits = type(self).needs_logits()

        self._inputs: typing.Dict[str, TritonInOut] = {}
        self._outputs: typing.Dict[str, TritonInOut] = {}

        self._triton_client: tritonclient.InferenceServerClient = None
        self._mem_pool: ResourcePool = None

        self._response_log_filename = response_log_filename
        self._response_log_dataframe: pd.DataFrame = None

        # Create the log DataFrame if requested
        if (self._response_log_filename is not None):

            self._response_log_dataframe = pd.DataFrame({"output": []})

        logger_grpc = logger.getChild("grpc")
        logger_grpc.addHandler(logging.FileHandler("grpc_triton.log", mode="w"))
        logger_grpc.propagate = False

        logger_http = logger.getChild("http")
        logger_http.addHandler(logging.FileHandler("http_triton.log", mode="w"))
        logger_http.propagate = False

    @classmethod
    def supports_cpp_node(cls):
        # Enable support by default
        return True

    @classmethod
    def needs_logits(cls):
        return False

    @classmethod
    def default_inout_mapping(cls) -> typing.Dict[str, str]:
        return {}

    def init(self):
        """
        This function instantiate triton client and memory allocation for inference input and output.

        """

        self._triton_client = tritonclient.InferenceServerClient(url=self._server_url, verbose=False)
        self._triton_client_http = tritonclient_http.InferenceServerClient(url=self._server_url[0:-1] + "0",
                                                                           verbose=False)

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
                raise RuntimeError("Mismatched Sequence Length. Config specified {} but model specified {}".format(
                    self._fea_length, int(model_meta["inputs"][0]["shape"][-1])))

            # Check batch size
            if (model_config.get("max_batch_size", 0) != self._max_batch_size):

                # If the model is more, thats fine. Gen warning
                if (model_config["max_batch_size"] > self._max_batch_size):
                    warnings.warn(("Model max batch size ({}) is more than configured max batch size ({}). "
                                   "May result in sub optimal performance").format(model_config["max_batch_size"],
                                                                                   self._max_batch_size))

                # If the model is less, raise error. Cant send more to Triton than the max batch size
                if (model_config["max_batch_size"] < self._max_batch_size):
                    raise RuntimeError(
                        ("Model max batch size ({}) is less than configured max batch size ({}). "
                         "Reduce max batch size to be less than or equal to model max batch size.").format(
                             model_config["max_batch_size"], self._max_batch_size))

            shm_config = {}

            def build_inout(x: dict):
                b = np.dtype(triton_to_np_dtype(x["datatype"])).itemsize

                shape = []

                for y in x["shape"]:
                    y_int = int(y)

                    if (y_int == -1):
                        y_int = self._max_batch_size

                    shape.append(y_int)

                    b *= y_int

                mapped_name = x["name"] if x["name"] not in self._inout_mapping else self._inout_mapping[x["name"]]

                return TritonInOut(name=x["name"],
                                   bytes=b,
                                   datatype=x["datatype"],
                                   shape=shape,
                                   mapped_name=mapped_name)

            for x in model_meta["inputs"]:

                self._inputs[x["name"]] = build_inout(x)

            for x in model_meta["outputs"]:

                assert x["name"] not in self._inputs, "Input/Output names must be unique from eachother"

                self._outputs[x["name"]] = build_inout(x)

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
            logger.exception("Exception occurred while coordinating with Triton. Exception message: \n{}\n".format(ex),
                             exc_info=ex)
            raise ex

    def calc_output_dims(self, x: MultiInferenceMessage) -> typing.Tuple:
        return (x.count, self._outputs[list(self._outputs.keys())[0]].shape[1])

    @abstractmethod
    def _build_response(self, batch: MultiInferenceMessage, result: tritonclient.InferResult) -> ResponseMemory:
        pass

    def _infer_callback(self,
                        cb: typing.Callable[[ResponseMemory], None],
                        m: InputWrapper,
                        b: MultiInferenceMessage,
                        result: tritonclient.InferResult,
                        error: tritonclient.InferenceServerException):

        # If its an error, return that here
        if (error is not None):
            raise error

        request_id = int(result.get_response().id)

        if (self._response_log_dataframe is not None):

            assert request_id not in self._response_log_dataframe.index, "Duplicate request_id: {} detected".format(request_id)

            # Save the response and update the CSV. Double serialize to encode new lines
            self._response_log_dataframe.loc[int(result.get_response().id)] = json.dumps(
                MessageToJson(result.get_response(), preserving_proto_field_name=True, indent=2))

            # Sort by the index before writing
            self._response_log_dataframe.sort_index(axis=1, inplace=True)

            # Must write to the CSV each time since we dont know when we are done
            self._response_log_dataframe.to_csv(self._response_log_filename,
                                                sep=";",
                                                index_label="id",
                                                doublequote=False,
                                                quotechar="'")

        # Print the output
        logger_grpc = logger.getChild("grpc")

        if (logger_grpc.isEnabledFor(logging.DEBUG)):
            logger_grpc.debug("Request ID: {}".format(request_id))
            logger_grpc.debug(result.as_numpy("output"))

        # Build response
        response_mem = self._build_response(b, result)

        # Call the callback with the memory
        cb(response_mem)

        self._mem_pool.return_obj(m)

    def process(self, batch: MultiInferenceMessage, cb: typing.Callable[[ResponseMemory], None]):
        """
        This function sends batch of events as a requests to Triton inference server using triton client API.

        Parameters
        ----------
        batch : `morpheus.pipeline.messages.MultiInferenceMessage`
            Mini-batch of inference messages.
        cb : typing.Callable[[`morpheus.pipeline.messages.ResponseMemory`], None]
            Callback to set the values for the inference response.

        """
        mem: InputWrapper = self._mem_pool.borrow()

        request_id = _TritonInferenceWorker.__REQUEST_COUNTER.get_and_inc()

        logger_http = logger.getChild("http")

        # See if we are interested in logging the http requests
        if (logger_http.isEnabledFor(logging.DEBUG)):
            inputs_http: typing.List[tritonclient_http.InferInput] = [
                mem.build_input_http(input.name,
                                     batch.get_input(input.mapped_name),
                                     force_convert_inputs=self._force_convert_inputs)
                for input in self._inputs.values()
            ]

            outputs_http = [tritonclient_http.InferRequestedOutput(output.name) for output in self._outputs.values()]

            response_http = self._triton_client_http.infer(
                model_name=self._model_name,
                inputs=inputs_http,
                request_id=str(request_id),
                #  callback=partial(self._infer_callback, cb, mem, batch),
                outputs=outputs_http)

            logger_http.debug("Request ID: {}".format(request_id))
            logger_http.debug(response_http.as_numpy("output"))

        inputs: typing.List[tritonclient.InferInput] = [
            mem.build_input(input.name,
                            batch.get_input(input.mapped_name),
                            force_convert_inputs=self._force_convert_inputs) for input in self._inputs.values()
        ]

        outputs = [tritonclient.InferRequestedOutput(output.name) for output in self._outputs.values()]

        # Inference call
        self._triton_client.async_infer(model_name=self._model_name,
                                        request_id=str(request_id),
                                        inputs=inputs,
                                        callback=partial(self._infer_callback, cb, mem, batch),
                                        outputs=outputs)


class TritonInferenceNLP(_TritonInferenceWorker):
    """
    This class extends TritonInference to deal with scenario-specific NLP models inference requests like building
    response.

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
    force_convert_inputs : bool, default = False
        Whether or not to convert the inputs to the type specified by Triton. This will happen automatically if no
        data would be lost in the conversion (i.e., float -> double). Set this to True to convert the input even if
        data would be lost (i.e., double -> float).
    use_shared_memory : bool, default = False
        Whether or not to use CUDA Shared IPC Memory for transferring data to Triton. Using CUDA IPC reduces network
        transfer time but requires that Morpheus and Triton are located on the same machine.
    inout_mapping : typing.Dict[str, str]
        Dictionary used to map pipeline input/output names to Triton input/output names. Use this if the
        Morpheus names do not match the model.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def needs_logits(cls):
        return True

    @classmethod
    def default_inout_mapping(cls) -> typing.Dict[str, str]:
        # Some models use different names for the same thing. Set that here but allow user customization
        return {
            "attention_mask": "input_mask",
            "output": "probs",
        }

    def _build_response(self, batch: MultiInferenceMessage, result: tritonclient.InferResult) -> ResponseMemoryProbs:

        output = {output.mapped_name: result.as_numpy(output.name) for output in self._outputs.values()}

        if (self._needs_logits):
            output = {key: 1.0 / (1.0 + np.exp(-val)) for key, val in output.items()}

        mem = ResponseMemoryProbs(
            count=output["probs"].shape[0],
            probs=cp.array(output["probs"]),  # For now, only support one output
        )

        return mem


class TritonInferenceFIL(_TritonInferenceWorker):
    """
    This class extends `TritonInference` to deal with scenario-specific FIL models inference requests like
    building response.

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
    force_convert_inputs : bool, default = False
        Whether or not to convert the inputs to the type specified by Triton. This will happen automatically if no
        data would be lost in the conversion (i.e., float -> double). Set this to True to convert the input even if
        data would be lost (i.e., double -> float).
    use_shared_memory: bool, default = False
        Whether or not to use CUDA Shared IPC Memory for transferring data to Triton. Using CUDA IPC reduces network
        transfer time but requires that Morpheus and Triton are located on the same machine.
    inout_mapping : typing.Dict[str, str]
        Dictionary used to map pipeline input/output names to Triton input/output names. Use this if the
        Morpheus names do not match the model.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def default_inout_mapping(cls) -> typing.Dict[str, str]:
        # Some models use different names for the same thing. Set that here but allow user customization
        return {
            "output__0": "probs",
        }

    def _build_response(self, batch: MultiInferenceMessage, result: tritonclient.InferResult) -> ResponseMemoryProbs:

        output = {output.mapped_name: result.as_numpy(output.name) for output in self._outputs.values()}

        for key, val in output.items():
            if (len(val.shape) == 1):
                output[key] = np.expand_dims(val, 1)

        mem = ResponseMemoryProbs(
            count=output["probs"].shape[0],
            probs=cp.array(output["probs"]),  # For now, only support one output
        )

        return mem


class TritonInferenceAE(_TritonInferenceWorker):
    """
    This class extends `TritonInference` to deal with inference processing specific to the AutoEncoder.

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
    force_convert_inputs : bool, default = False
        Whether or not to convert the inputs to the type specified by Triton. This will happen automatically if no
        data would be lost in the conversion (i.e., float -> double). Set this to True to convert the input even if
        data would be lost (i.e., double -> float).
    use_shared_memory: bool, default = False
        Whether or not to use CUDA Shared IPC Memory for transferring data to Triton. Using CUDA IPC reduces network
        transfer time but requires that Morpheus and Triton are located on the same machine.
    inout_mapping : typing.Dict[str, str]
        Dictionary used to map pipeline input/output names to Triton input/output names. Use this if the
        Morpheus names do not match the model.

    """

    def __init__(self, inf_queue: ProducerConsumerQueue, c: Config, *args, **kwargs):
        super().__init__(inf_queue=inf_queue, c=c, *args, **kwargs)

        import torch
        from dfencoder import AutoEncoder

        # Save the autoencoder path
        with open(c.ae.autoencoder_path, 'rb') as in_strm:
            self._autoencoder = AutoEncoder()
            self._autoencoder.load_state_dict(torch.load(in_strm))

            # Ensure that there is a label_smoothing property on cce. Necessary if pytorch version is different
            if (not hasattr(self._autoencoder.cce, "label_smoothing")):
                self._autoencoder.cce.label_smoothing = 0.0

    @classmethod
    def supports_cpp_node(cls):
        # Enable support by default
        return False

    def _build_response(self, batch: MultiInferenceMessage, result: tritonclient.InferResult) -> ResponseMemoryProbs:

        import torch

        output = {output.mapped_name: result.as_numpy(output.name) for output in self._outputs.values()}

        data = self._autoencoder.prepare_df(batch.get_meta())
        num_target, bin_target, codes = self._autoencoder.compute_targets(data)
        mse_loss = self._autoencoder.mse(torch.as_tensor(output["num"], device='cuda'), num_target)
        net_loss = [mse_loss.data]
        bce_loss = self._autoencoder.bce(torch.as_tensor(output["bin"], device='cuda'), bin_target)
        net_loss += [bce_loss.data]
        cce_loss = []
        for i, ft in enumerate(self._autoencoder.categorical_fts):
            loss = self._autoencoder.cce(torch.as_tensor(output[ft], device='cuda'), codes[i])
            cce_loss.append(loss)
            net_loss += [loss.data.reshape(-1, 1)]

        net_loss = torch.cat(net_loss, dim=1).mean(dim=1)
        ae_scores = cp.asarray(net_loss)
        ae_scores = ae_scores.reshape((batch.count, 1))

        mem = ResponseMemoryProbs(
            count=batch.count,
            probs=ae_scores,  # For now, only support one output
        )

        return mem


@register_stage("inf-triton")
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
    response_log_filename: str, default = None
        Saves the responses from Triton into a CSV file to support debugging and mocking. Use `None` to disable this
        feature. Requires the stage to be run in python mode.
    """

    def __init__(self,
                 c: Config,
                 model_name: str,
                 server_url: str,
                 force_convert_inputs: bool = False,
                 use_shared_memory: bool = False,
                 response_log_filename: str = None):
        super().__init__(c)

        self._config = c

        self._kwargs = {
            "model_name": model_name,
            "server_url": server_url,
            "force_convert_inputs": force_convert_inputs,
            "use_shared_memory": use_shared_memory,
        }

        if (response_log_filename is not None):
            if (self._build_cpp_node()):
                logger.warning("Cannot set response_log_filename when using C++ stages")
            else:
                self._kwargs["response_log_filename"] = response_log_filename

        self._requires_seg_ids = False

    def supports_cpp_node(self):
        # Get the value from the worker class
        return self._get_worker_class().supports_cpp_node()

    def _get_worker_class(self):
        if (self._config.mode == PipelineModes.NLP):
            return TritonInferenceNLP
        elif (self._config.mode == PipelineModes.FIL):
            return TritonInferenceFIL
        elif (self._config.mode == PipelineModes.AE):
            return TritonInferenceAE
        else:
            raise NotImplementedError("Unknown config mode")

    def _get_inference_worker(self, inf_queue: ProducerConsumerQueue) -> InferenceWorker:

        worker_cls = self._get_worker_class()

        return worker_cls(inf_queue=inf_queue, c=self._config, **self._kwargs)

    def _get_cpp_inference_node(self, builder: srf.Builder):

        return _stages.InferenceClientStage(builder,
                                            name=self.unique_name,
                                            needs_logits=self._get_worker_class().needs_logits(),
                                            inout_mapping=self._get_worker_class().default_inout_mapping(),
                                            **self._kwargs)
