# Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

import numpy as np

from morpheus.messages import MessageMeta
from morpheus_llm.llm import LLMContext
from morpheus_llm.llm import LLMNodeBase

logger = logging.getLogger(__name__)


def _array_to_list(array):
    """
    Convert all numpy arrays to lists. Recursive.
    """
    if isinstance(array, np.ndarray):
        return _array_to_list(array.tolist())

    if isinstance(array, list):
        return [_array_to_list(item) for item in array]

    if isinstance(array, dict):
        return {key: _array_to_list(value) for key, value in array.items()}

    if isinstance(array, tuple):
        return tuple(_array_to_list(item) for item in array)

    return array


class ExtracterNode(LLMNodeBase):
    """
    Extracts fields from the DataFrame contained by the message attached to the `LLMContext` and copies them directly
    to the context.

    The list of fields to be extracted is provided by the task's `input_keys` attached to the `LLMContext`.
    """

    def get_input_names(self) -> list[str]:
        # This node does not receive its inputs from upstream nodes, but rather from the task itself
        return []

    async def execute(self, context: LLMContext) -> LLMContext:  # pylint: disable=invalid-overridden-method

        # Get the keys from the task
        input_keys: list[str] = typing.cast(list[str], context.task()["input_keys"])

        meta: MessageMeta = context.message().get_metadata("llm_message_meta")

        with meta.mutable_dataframe() as df:
            input_dict: list[dict] = df[input_keys].to_dict(orient="list")

        input_dict = _array_to_list(input_dict)

        if (len(input_keys) == 1):
            # Extract just the first key if there is only 1
            context.set_output(input_dict[input_keys[0]])
        else:
            context.set_output(input_dict)

        return context


class ManualExtracterNode(LLMNodeBase):
    """
    Extracts fields from the DataFrame contained by the message attached to the `LLMContext` and copies them directly
    to the context.

    The list of fields to be extracted is manually specified when the node is constructed.
    """

    def __init__(self, input_names: list[str]) -> None:
        super().__init__()

        assert len(input_names) > 0, "At least one input name must be provided"
        assert len(set(input_names)) == len(input_names), "Input names must be unique"

        self._input_names = input_names

    def get_input_names(self) -> list[str]:
        return self._input_names

    async def execute(self, context: LLMContext) -> LLMContext:  # pylint: disable=invalid-overridden-method

        # Get the data from the DataFrame
        meta: MessageMeta = context.message().get_metadata("llm_message_meta")
        with meta.mutable_dataframe() as df:
            input_dict: list[dict] = df[self._input_names].to_dict(orient="list")

        input_dict = _array_to_list(input_dict)

        if (len(self._input_names) == 1):
            # Extract just the first key if there is only 1
            context.set_output(input_dict[self._input_names[0]])
        else:
            context.set_output(input_dict)

        return context
