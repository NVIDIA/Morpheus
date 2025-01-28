# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus_llm.llm import LLMContext
from morpheus_llm.llm import LLMTaskHandler

logger = logging.getLogger(__name__)


class SimpleTaskHandler(LLMTaskHandler):
    """
    Copies fields from an `LLMContext` to columns in the DataFrame contained in the `ControlMessage` payload.

    Parameters
    ----------
    output_columns : list[str], optional
        The list of columns to copy from the `LLMContext` instance to the DataFrame. If `None`, defaults to
        `["response"]`.
    """

    def __init__(self, output_columns: list[str] = None) -> None:
        super().__init__()

        if (output_columns is None):
            self._output_columns = ["response"]
        else:
            self._output_columns = output_columns

    def get_input_names(self) -> list[str]:
        return self._output_columns

    # pylint: disable=invalid-overridden-method
    async def try_handle(self, context: LLMContext) -> list[ControlMessage]:

        input_dict = context.get_inputs()

        meta: MessageMeta = context.message().get_metadata("llm_message_meta")
        with meta.mutable_dataframe() as df:
            # Write the values to the dataframe
            for key, value in input_dict.items():
                df[key] = value

        return [context.message()]
