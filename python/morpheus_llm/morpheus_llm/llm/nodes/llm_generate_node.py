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
import typing

from morpheus_llm.llm import LLMContext
from morpheus_llm.llm import LLMNodeBase
from morpheus_llm.llm.services.llm_service import LLMClient

logger = logging.getLogger(__name__)


class LLMGenerateNode(LLMNodeBase):
    """
    Generates responses from an LLM using the provided `llm_client` instance based on prompts provided as input from
    upstream nodes.
    """

    def __init__(self, llm_client: LLMClient, return_exceptions=False) -> None:
        """
        Parameters
        ----------
        llm_client: LLMClient
            The client instance to use to generate responses.
        return_exceptions: bool, optional
            Whether to return exceptions when generating responses. If set to False, the first exception that is
            returned will stop all other items in the batch from processing Defaults to `False`.
        """

        super().__init__()

        self._llm_client = llm_client
        self._return_exceptions: typing.Literal[True] | typing.Literal[False] = return_exceptions

    def get_input_names(self) -> list[str]:
        return self._llm_client.get_input_names()

    async def execute(self, context: LLMContext) -> LLMContext:  # pylint: disable=invalid-overridden-method

        # Get the inputs
        inputs: dict[str, list[str]] = context.get_inputs()  # type: ignore

        results = await self._llm_client.generate_batch_async(inputs, return_exceptions=self._return_exceptions)

        context.set_output(results)

        return context
