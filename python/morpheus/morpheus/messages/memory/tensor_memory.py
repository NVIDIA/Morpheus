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
import typing

import morpheus._lib.messages as _messages
from morpheus.messages.message_base import MessageData
from morpheus.utils.type_aliases import NDArrayType
from morpheus.utils.type_aliases import TensorMapType


@dataclasses.dataclass(init=False)
class TensorMemory(MessageData, cpp_class=_messages.TensorMemory):
    """
    This is a base container class for data that will be used for inference stages. This class is designed to
    hold generic tensor data in either CuPy or NumPy arrays.

    Parameters
    ----------
    count : int
        Length of each tensor contained in `tensors`.
    tensors : TensorMapType
        Collection of tensors uniquely identified by a name.

    """
    count: int
    tensors: TensorMapType = dataclasses.field(repr=False)

    def __init__(self, *, count: int = None, tensors: TensorMapType = None):

        self.count = count

        if tensors is None:
            tensors = {}
        else:
            self._check_tensors(tensors)

        self._tensors = tensors

    def _check_tensors(self, tensors: TensorMapType):
        for tensor in tensors.values():
            self._check_tensor(tensor)

    def _check_tensor(self, tensor: NDArrayType):
        if (tensor.shape[0] != self.count):
            class_name = type(self).__name__
            raise ValueError(
                f"The number rows in tensor {tensor.shape[0]} does not match {class_name}.count of {self.count}")

    def __getattr__(self, name: str) -> typing.Any:
        if ("tensors" in self.__dict__ and self.has_tensor(name)):
            return self.get_tensor(name)

        if hasattr(super(), "__getattr__"):
            return super().__getattr__(name)
        raise AttributeError

    @property
    def tensor_names(self) -> typing.List[str]:
        return list(self._tensors.keys())

    def has_tensor(self, name: str) -> bool:
        """
        Returns True if a tensor with the requested name exists in the tensors object

        Parameters
        ----------
        name : str
            Name to lookup

        Returns
        -------
        bool
            True if the tensor was found
        """
        return name in self._tensors

    def get_tensors(self):
        """
        Get the tensors contained by this instance. It is important to note that when C++ execution is enabled the
        returned tensors will be a Python copy of the tensors stored in the C++ object. As such any changes made to the
        tensors will need to be updated with a call to `set_tensors`.

        Returns
        -------
        TensorMapType
        """
        return self._tensors

    def set_tensors(self, tensors: TensorMapType):
        """
        Overwrite the tensors stored by this instance. If the length of the tensors has changed, then the `count`
        property should also be updated.

        Parameters
        ----------
        tensors : TensorMapType
            Collection of tensors uniquely identified by a name.
        """
        self._check_tensors(tensors)
        self._tensors = tensors

    def get_tensor(self, name: str):
        """
        Get the Tensor stored in the container identified by `name`.

        Parameters
        ----------
        name : str
            Tensor key name.

        Returns
        -------
        NDArrayType
            Tensor.

        Raises
        ------
        KeyError
            If tensor name does not exist in the container.
        """
        return self._tensors[name]

    def _get_tensor_prop(self, name: str):
        """
        This method is intended to be used by propery methods in subclasses

        Parameters
        ----------
        name : str
            Tensor key name.

        Returns
        -------
        NDArrayType
            Tensor.

        Raises
        ------
        AttributeError
            If tensor name does not exist in the container.
        """
        try:
            return self._tensors[name]
        except KeyError as e:
            raise AttributeError from e

    def set_tensor(self, name: str, tensor: NDArrayType):
        """
        Update the tensor identified by `name`.

        Parameters
        ----------
        name : str
            Tensor key name.
        tensor : NDArrayType
            Tensor as either a CuPy or NumPy array.

        Raises
        ------
        ValueError
            If the number of rows in `tensor` does not match `count`
        """
        self._check_tensor(tensor)
        self._tensors[name] = tensor
