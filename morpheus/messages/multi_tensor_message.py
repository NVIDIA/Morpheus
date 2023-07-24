# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from morpheus.messages.memory.tensor_memory import TensorMemory
from morpheus.messages.message_meta import MessageMeta
from morpheus.messages.multi_message import MultiMessage

# Needed to provide the return type of `@classmethod`
Self = typing.TypeVar("Self", bound="MultiTensorMessage")


@dataclasses.dataclass
class MultiTensorMessage(_messages.MultiTensorMessage):
    """
    This class contains several inference responses as well as the cooresponding message metadata.

    Parameters
    ----------
    memory : `TensorMemory`
        Container holding generic tensor data in cupy arrays
    offset : int
        Offset of each message into the `TensorMemory` block.
    count : int
        Number of rows in the `TensorMemory` block.
    """

    required_tensors: typing.ClassVar[typing.List[str]] = []
    """The tensor names that are required for instantiation"""
    id_tensor_name: typing.ClassVar[str] = "seq_ids"
    """Name of the tensor that correlates tensor rows to message IDs"""

    def __init__(self,
                 *,
                 meta: MessageMeta,
                 mess_offset: int = 0,
                 mess_count: int = -1,
                 memory: TensorMemory,
                 offset: int = 0,
                 count: int = -1,
                 id_tensor_name: str = "seq_ids"):

        if memory is None:
            raise ValueError(f"Must define `memory` when creating {self.__class__.__name__}")

        # Use the meta count if not supplied
        if (count == -1):
            count = memory.count - offset

        # Check for valid offsets and counts
        if offset < 0 or offset >= memory.count:
            raise ValueError("Invalid offset value")
        if count <= 0 or (offset + count > memory.count):
            raise ValueError("Invalid count value")

        # Call the base class last because the properties need to be initialized first
        super().__init__(
            meta=meta,
            mess_offset=mess_offset,
            mess_count=mess_count,
            memory=memory,
            offset=offset,
            count=count,
            id_tensor_name=id_tensor_name)


        if (self.count < self.mess_count):
            raise ValueError("Invalid count value. Must have a count greater than or equal to mess_count")

        # Check the ID tensor for consistency
        self._check_id_tensor()

        # Finally, check for the required tensors class attribute
        if (hasattr(self.__class__, "required_tensors")):
            for tensor_name in self.__class__.required_tensors:
                if (not memory.has_tensor(tensor_name)):
                    raise ValueError((f"`TensorMemory` object must have a '{tensor_name}' "
                                      f"tensor to create `{self.__class__.__name__}`").format(self.__class__.__name__))

    @property
    def tensors(self):
        """
        Get tensors stored in the TensorMemory container sliced according to `offset` and `count`.

        Returns
        -------
        cupy.ndarray
            Inference tensors.

        """
        tensors = self.memory.get_tensors()
        return {key: self.get_tensor(key) for key in tensors.keys()}

    def __getattr__(self, name: str) -> typing.Any:
        if ("memory" in self.__dict__ and self.memory.has_tensor(name)):
            return self.get_tensor(name)

        if hasattr(super(), "__getattr__"):
            return super().__getattr__(name)
        raise AttributeError

    def _check_id_tensor(self):

        if (self.memory.has_tensor(self.id_tensor_name)):
            # Check the bounds against the elements in the array
            id_tensor = self.memory.get_tensor(self.id_tensor_name)

            first_element = id_tensor[self.offset, 0].item()
            last_element = id_tensor[self.offset + self.count - 1, 0].item()

            if (first_element != self.mess_offset):
                raise RuntimeError(f"Inconsistent ID column. First element in '{self.id_tensor_name}' tensor, "
                                   f"[{first_element}], must match mess_offset, [{self.mess_offset}]")

            if (last_element != self.mess_offset + self.mess_count - 1):
                raise RuntimeError(f"Inconsistent ID column. Last element in '{self.id_tensor_name}' tensor, "
                                   f"[{last_element}], must not extend beyond last message, "
                                   f"[{self.mess_offset + self.mess_count - 1}]")

    def copy_tensor_ranges(self, ranges, mask=None):
        """
        Perform a copy of the underlying tensor tensors for the given `ranges` of rows.

        Parameters
        ----------
        ranges : typing.List[typing.Tuple[int, int]]
            Rows to include in the copy in the form of `[(`start_row`, `stop_row`),...]`
            The `stop_row` isn't included. For example to copy rows 1-2 & 5-7 `ranges=[(1, 3), (5, 8)]`

        mask : typing.Union[None, cupy.ndarray, numpy.ndarray]
            Optionally specify rows as a cupy array (when using cudf Dataframes) or a numpy array (when using pandas
            Dataframes) of booleans. When not-None `ranges` will be ignored. This is useful as an optimization as this
            avoids needing to generate the mask on it's own.

        Returns
        -------
        typing.Dict[str, cupy.ndarray]
        """
        if mask is None:
            mask = self._ranges_to_mask(self.get_meta(), ranges=ranges)

        # The tensors property method returns a copy with the offsets applied
        tensors = self.tensors
        return {key: tensor[mask] for (key, tensor) in tensors.items()}

    def copy_ranges(self, ranges: typing.List[typing.Tuple[int, int]]):
        """
        Perform a copy of the current message, dataframe and tensors for the given `ranges` of rows.

        Parameters
        ----------
        ranges : typing.List[typing.Tuple[int, int]]
            Rows to include in the copy in the form of `[(`start_row`, `stop_row`),...]`
            The `stop_row` isn't included. For example to copy rows 1-2 & 5-7 `ranges=[(1, 3), (5, 8)]`

        -------
        `MultiTensorMessage`
        """
        mask = self._ranges_to_mask(self.get_meta(), ranges)
        sliced_rows = self.copy_meta_ranges(ranges, mask=mask)
        sliced_count = len(sliced_rows)
        sliced_tensors = self.copy_tensor_ranges(ranges, mask=mask)

        mem = TensorMemory(count=sliced_count, tensors=sliced_tensors)

        return self.from_message(self,
                                 meta=MessageMeta(sliced_rows),
                                 mess_offset=0,
                                 mess_count=sliced_count,
                                 memory=mem,
                                 offset=0,
                                 count=sliced_count)
