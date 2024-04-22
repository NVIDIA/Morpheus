# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

import asyncio
import logging
import os
import typing

from morpheus.llm.services.llm_service import LLMClient
from morpheus.llm.services.llm_service import LLMService

logger = logging.getLogger(__name__)

IMPORT_EXCEPTION = None
IMPORT_ERROR_MESSAGE = (
    "The `langchain-nvidia-ai-endpoints` package was not found. Install it and other additional dependencies by running the following command:\n"
    "`conda env update --solver=libmamba -n morpheus "
    "--file morpheus/conda/environments/dev_cuda-121_arch-x86_64.yaml --prune`")

try:
    from langchain_core.prompt_values import StringPromptValue
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    from langchain_nvidia_ai_endpoints._common import NVEModel
except ImportError as import_exc:
    IMPORT_EXCEPTION = import_exc


class NVFoundationLLMClient(LLMClient):
    """
    Client for interacting with a specific model in Nemo. This class should be constructed with the
    `NeMoLLMService.get_client` method.

    Parameters
    ----------
    parent : NeMoLLMService
        The parent service for this client.
    model_name : str
        The name of the model to interact with.

    model_kwargs : dict[str, typing.Any]
        Additional keyword arguments to pass to the model when generating text.
    """

    def __init__(self, parent: "NVFoundationLLMService", *, model_name: str, **model_kwargs) -> None:
        if IMPORT_EXCEPTION is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE) from IMPORT_EXCEPTION

        super().__init__()

        assert parent is not None, "Parent service cannot be None."

        self._parent = parent
        self._model_name = model_name
        self._model_kwargs = model_kwargs
        self._prompt_key = "prompt"

        self._client = ChatNVIDIA(client=self._parent._nve_client, model=model_name, **model_kwargs)

    def get_input_names(self) -> list[str]:
        schema = self._client.get_input_schema()

        return [self._prompt_key]

    def generate(self, **input_dict) -> str:
        """
        Issue a request to generate a response based on a given prompt.

        Parameters
        ----------
        input_dict : dict
            Input containing prompt data.
        """
        return self.generate_batch({self._prompt_key: [input_dict[self._prompt_key]]})[0]

    async def generate_async(self, **input_dict) -> str:
        """
        Issue an asynchronous request to generate a response based on a given prompt.

        Parameters
        ----------
        input_dict : dict
            Input containing prompt data.
        """

        inputs = {self._prompt_key: [input_dict[self._prompt_key]]}

        input_dict.pop(self._prompt_key)

        return (await self.generate_batch_async(inputs=inputs, **input_dict))[0]

    def generate_batch(self, inputs: dict[str, list], **kwargs) -> list[str]:
        """
        Issue a request to generate a list of responses based on a list of prompts.

        Parameters
        ----------
        inputs : dict
            Inputs containing prompt data.
        """
        prompts = [StringPromptValue(text=p) for p in inputs[self._prompt_key]]

        responses = self._client.generate_prompt(prompts=prompts, **self._model_kwargs)  # type: ignore

        return [g[0].text for g in responses.generations]

    async def generate_batch_async(self, inputs: dict[str, list], **kwargs) -> list[str]:
        """
        Issue an asynchronous request to generate a list of responses based on a list of prompts.

        Parameters
        ----------
        inputs : dict
            Inputs containing prompt data.
        """

        prompts = [StringPromptValue(text=p) for p in inputs[self._prompt_key]]

        final_kwargs = {**self._model_kwargs, **kwargs}

        responses = await self._client.agenerate_prompt(prompts=prompts, **final_kwargs)  # type: ignore

        return [g[0].text for g in responses.generations]


class NVFoundationLLMService(LLMService):
    """
    A service for interacting with NeMo LLM models, this class should be used to create a client for a specific model.

    Parameters
    ----------
    api_key : str, optional
        The API key for the LLM service, by default None. If `None` the API key will be read from the `NGC_API_KEY`
        environment variable. If neither are present an error will be raised.

    org_id : str, optional
        The organization ID for the LLM service, by default None. If `None` the organization ID will be read from the
        `NGC_ORG_ID` environment variable. This value is only required if the account associated with the `api_key` is
        a member of multiple NGC organizations.
    """

    def __init__(self, *, api_key: str = None) -> None:
        if IMPORT_EXCEPTION is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE) from IMPORT_EXCEPTION

        super().__init__()

        self._api_key = api_key

        self._nve_client = NVEModel(
            nvidia_api_key=self._api_key,
            fetch_url_format=f"{os.getenv('NVIDIA_API_BASE', 'https://api.nvcf.nvidia.com/v2')}/nvcf/pexec/status/",
            call_invoke_base=f"{os.getenv('NVIDIA_API_BASE', 'https://api.nvcf.nvidia.com/v2')}/nvcf/pexec/functions",
            func_list_format=f"{os.getenv('NVIDIA_API_BASE', 'https://api.nvcf.nvidia.com/v2')}/nvcf/functions",
        )  # type: ignore

    def get_client(self, *, model_name: str, **model_kwargs) -> NVFoundationLLMClient:
        """
        Returns a client for interacting with a specific model. This method is the preferred way to create a client.

        Parameters
        ----------
        model_name : str
            The name of the model to create a client for.

        model_kwargs : dict[str, typing.Any]
            Additional keyword arguments to pass to the model when generating text.
        """

        return NVFoundationLLMClient(self, model_name=model_name, **model_kwargs)
