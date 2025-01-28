# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

import typing

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

from morpheus_llm.llm.services.llm_service import LLMClient


class LangchainLLMClientWrapper(LLM):
    """
    Wrapper for the LLMClient to be used in the LLMEngine.

    Parameters
    ----------
    client : LLMClient
        The client to use to run the LLM.
    """

    # Disable the unused-argument warning, we want to keep the same function signature as the parent, however in CI
    # langchain is not installed which triggers a false positive warning.
    # pylint: disable=unused-argument

    client: LLMClient

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "morpheus"

    def _call(
        self,
        prompt: str,
        stop: typing.Optional[list[str]] = None,
        run_manager: typing.Optional[CallbackManagerForLLMRun] = None,
        **kwargs: typing.Any,
    ) -> str:
        """Run the LLM on the given prompt and input."""

        return self.client.generate(prompt=prompt, stop=stop)

    async def _acall(
        self,
        prompt: str,
        stop: typing.Optional[list[str]] = None,
        run_manager: typing.Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: typing.Any,
    ) -> str:
        """Run the LLM on the given prompt and input."""
        return await self.client.generate_async(prompt=prompt, stop=stop)
