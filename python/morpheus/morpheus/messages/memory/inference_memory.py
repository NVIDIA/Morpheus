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

import dataclasses

import morpheus._lib.messages as _messages
from morpheus.messages.data_class_prop import DataClassProp
from morpheus.messages.memory.tensor_memory import TensorMemory
from morpheus.utils.type_aliases import NDArrayType


@dataclasses.dataclass(init=False)
class InferenceMemory(TensorMemory, cpp_class=_messages.InferenceMemory):
    """
    This is a base container class for data that will be used for inference stages. This class is designed to
    hold generic tensor data in either CuPy or NumPy arrays.
    """

    def get_input(self, name: str):
        """
        Get the tensor stored in the container identified by `name`. Alias for `InferenceMemory.get_tensor`.

        Parameters
        ----------
        name : str
            Key used to do lookup in inputs dict of the container.

        Returns
        -------
        NDArrayType
            Inputs corresponding to name.

        Raises
        ------
        KeyError
            If input name does not exist in the container.
        """
        return self.get_tensor(name)

    def set_input(self, name: str, tensor: NDArrayType):
        """
        Update the input tensor identified by `name`. Alias for `InferenceMemory.set_tensor`

        Parameters
        ----------
        name : str
            Key used to do lookup in inputs dict of the container.
        tensor : NDArrayType
            Tensor as either CuPy or NumPy array.
        """
        self.set_tensor(name, tensor)


@dataclasses.dataclass(init=False)
class InferenceMemoryNLP(InferenceMemory, cpp_class=_messages.InferenceMemoryNLP):
    """
    This is a container class for data that needs to be submitted to the inference server for NLP category
    usecases.

    Parameters
    ----------
    input_ids : NDArrayType
        The token-ids for each string padded with 0s to max_length.
    input_mask : NDArrayType
        The mask for token-ids result where corresponding positions identify valid token-id values.
    seq_ids : NDArrayType
        Ids used to index from an inference input to a message. Necessary since there can be more inference
        inputs than messages (i.e., if some messages get broken into multiple inference requests).

    """
    input_ids: dataclasses.InitVar[NDArrayType] = DataClassProp(InferenceMemory._get_tensor_prop,
                                                                InferenceMemory.set_input)
    input_mask: dataclasses.InitVar[NDArrayType] = DataClassProp(InferenceMemory._get_tensor_prop,
                                                                 InferenceMemory.set_input)
    seq_ids: dataclasses.InitVar[NDArrayType] = DataClassProp(InferenceMemory._get_tensor_prop,
                                                              InferenceMemory.set_input)

    def __init__(self, *, count: int, input_ids: NDArrayType, input_mask: NDArrayType, seq_ids: NDArrayType):
        super().__init__(count=count, tensors={'input_ids': input_ids, 'input_mask': input_mask, 'seq_ids': seq_ids})


@dataclasses.dataclass(init=False)
class InferenceMemoryFIL(InferenceMemory, cpp_class=_messages.InferenceMemoryFIL):
    """
    This is a container class for data that needs to be submitted to the inference server for FIL category
    usecases.

    Parameters
    ----------
    input__0 : NDArrayType
        Inference input.
    seq_ids : NDArrayType
        Ids used to index from an inference input to a message. Necessary since there can be more inference
        inputs than messages (i.e., if some messages get broken into multiple inference requests).

    """
    input__0: dataclasses.InitVar[NDArrayType] = DataClassProp(InferenceMemory._get_tensor_prop,
                                                               InferenceMemory.set_input)
    seq_ids: dataclasses.InitVar[NDArrayType] = DataClassProp(InferenceMemory._get_tensor_prop,
                                                              InferenceMemory.set_input)

    def __init__(self, *, count: int, input__0: NDArrayType, seq_ids: NDArrayType):
        super().__init__(count=count, tensors={'input__0': input__0, 'seq_ids': seq_ids})


@dataclasses.dataclass(init=False)
class InferenceMemoryAE(InferenceMemory, cpp_class=None):
    """
    This is a container class for data that needs to be submitted to the inference server for auto encoder usecases.

    Parameters
    ----------
    inputs : NDArrayType
        Inference input.
    seq_ids : NDArrayType
        Ids used to index from an inference input to a message. Necessary since there can be more inference
        inputs than messages (i.e., if some messages get broken into multiple inference requests).
    """

    input: dataclasses.InitVar[NDArrayType] = DataClassProp(InferenceMemory._get_tensor_prop, InferenceMemory.set_input)
    seq_ids: dataclasses.InitVar[NDArrayType] = DataClassProp(InferenceMemory._get_tensor_prop,
                                                              InferenceMemory.set_input)

    def __init__(self, *, count: int, inputs: NDArrayType, seq_ids: NDArrayType):
        super().__init__(count=count, tensors={'input': inputs, 'seq_ids': seq_ids})
