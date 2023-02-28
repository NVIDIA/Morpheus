# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cupy as cp

import morpheus._lib.messages as _messages
from morpheus.messages.memory.inference_memory import InferenceMemory
from morpheus.messages.multi_message import MultiMessage


@dataclasses.dataclass
class MultiInferenceMessage(MultiMessage, cpp_class=_messages.MultiInferenceMessage):
    """
    This is a container class that holds the InferenceMemory container and the metadata of the data contained
    within it. Builds on top of the `MultiMessage` class to add additional data for inferencing.

    This class requires two separate memory blocks for a batch. One for the message metadata (i.e., start time,
    IP address, etc.) and another for the raw inference inputs (i.e., input_ids, seq_ids). Since there can be
    more inference input requests than messages (This happens when some messages get broken into multiple
    inference requests) this class stores two different offset and count values. `mess_offset` and
    `mess_count` refer to the offset and count in the message metadata batch and `offset` and `count` index
    into the inference batch data.

    Parameters
    ----------
    memory : `InferenceMemory`
        Inference memory.
    offset : int
        Message offset in inference memory instance.
    count : int
        Message count in inference memory instance.

    """
    memory: InferenceMemory = dataclasses.field(repr=False)
    offset: int
    count: int

    @property
    def inputs(self):
        """
        Get inputs stored in the InferenceMemory container.

        Returns
        -------
        cupy.ndarray
            Inference inputs.

        """
        tensors = self.memory.get_tensors()
        return {key: self.get_input(key) for key in tensors.keys()}

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d

    def __getattr__(self, name: str) -> typing.Any:
        return self.get_input(name)

    def get_input(self, name: str):
        """
        Get input stored in the InferenceMemory container.

        Parameters
        ----------
        name : str
            Input key name.

        Returns
        -------
        cupy.ndarray
            Inference input.

        Raises
        ------
        AttributeError
            When no matching input tensor exists.
        """
        return self.memory.get_input(name)[self.offset:self.offset + self.count, :]

    def get_slice(self, start, stop):
        """
        Returns sliced batches based on offsets supplied. Automatically calculates the correct `mess_offset`
        and `mess_count`.

        Parameters
        ----------
        start : int
            Start offset address.
        stop : int
            Stop offset address.

        Returns
        -------
        `MultiInferenceMessage`
            A new `MultiInferenceMessage` with sliced offset and count.

        """
        mess_start = self.mess_offset + self.seq_ids[start, 0].item()
        mess_stop = self.mess_offset + self.seq_ids[stop - 1, 0].item() + 1
        return MultiInferenceMessage(meta=self.meta,
                                     mess_offset=mess_start,
                                     mess_count=mess_stop - mess_start,
                                     memory=self.memory,
                                     offset=start,
                                     count=stop - start)


@dataclasses.dataclass
class MultiInferenceNLPMessage(MultiInferenceMessage, cpp_class=_messages.MultiInferenceNLPMessage):
    """
    A stronger typed version of `MultiInferenceMessage` that is used for NLP workloads. Helps ensure the
    proper inputs are set and eases debugging.
    """

    @property
    def input_ids(self):
        """
        Returns token-ids for each string padded with 0s to max_length.

        Returns
        -------
        cupy.ndarray
            The token-ids for each string padded with 0s to max_length.

        """

        return self.get_input("input_ids")

    @property
    def input_mask(self):
        """
        Returns mask for token-ids result where corresponding positions identify valid token-id values.

        Returns
        -------
        cupy.ndarray
            The mask for token-ids result where corresponding positions identify valid token-id values.

        """

        return self.get_input("input_mask")

    @property
    def seq_ids(self):
        """
        Returns sequence ids, which are used to keep track of which inference requests belong to each message.

        Returns
        -------
        cupy.ndarray
            Ids used to index from an inference input to a message. Necessary since there can be more
            inference inputs than messages (i.e., if some messages get broken into multiple inference requests).

        """

        return self.get_input("seq_ids")


@dataclasses.dataclass
class MultiInferenceFILMessage(MultiInferenceMessage, cpp_class=_messages.MultiInferenceFILMessage):
    """
    A stronger typed version of `MultiInferenceMessage` that is used for FIL workloads. Helps ensure the
    proper inputs are set and eases debugging.
    """

    @property
    def input__0(self):
        """
        Input to FIL model inference.

        Returns
        -------
        cupy.ndarray
            Input data.

        """

        return self.get_input("input__0")

    @property
    def seq_ids(self):
        """
        Returns sequence ids, which are used to keep track of messages in a multi-threaded environment.

        Returns
        -------
        cupy.ndarray
            Sequence ids.

        """

        return self.get_input("seq_ids")
