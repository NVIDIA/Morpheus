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
import warnings

from morpheus.utils.env_config_value import EnvConfigValue
from morpheus_llm.error import IMPORT_ERROR_MESSAGE
from morpheus_llm.llm.services.llm_service import LLMClient
from morpheus_llm.llm.services.llm_service import LLMService

logger = logging.getLogger(__name__)

IMPORT_EXCEPTION = None

try:
    from langchain_core.prompt_values import StringPromptValue
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
except ImportError as import_exc:
    IMPORT_EXCEPTION = import_exc


class NVFoundationLLMClient(LLMClient):
    """
    Client for interacting with a specific model in Nemo. This class should be constructed with the
    `NeMoLLMService.get_client` method.

    Parameters
    ----------
    parent : NVFoundationMService
        The parent service for this client.
    model_name : str
        The name of the model to interact with.
    model_kwargs : dict[str, typing.Any]
        Additional keyword arguments to pass to the model when generating text.
    """

    def __init__(self, parent: "NVFoundationLLMService", *, model_name: str, **model_kwargs) -> None:
        if IMPORT_EXCEPTION is not None:
            raise ImportError(
                IMPORT_ERROR_MESSAGE.format(package='langchain-nvidia-ai-endpoints')) from IMPORT_EXCEPTION

        super().__init__()

        assert parent is not None, "Parent service cannot be None."

        self._parent = parent
        self._model_name = model_name
        self._model_kwargs = model_kwargs
        self._prompt_key = "prompt"

        chat_kwargs = {
            "model": model_name,
            "api_key": self._parent._api_key.value,
            "base_url": self._parent._base_url.value,
        }

        # Remove None values set by the environment in the kwargs
        if (chat_kwargs["api_key"] is None):
            del chat_kwargs["api_key"]

        if (chat_kwargs["base_url"] is None):
            del chat_kwargs["base_url"]

        # Combine the chat args with the model
        self._client = ChatNVIDIA(**{**chat_kwargs, **model_kwargs})  # type: ignore

    def get_input_names(self) -> list[str]:
        return [self._prompt_key]

    @property
    def model_kwargs(self):
        return self._model_kwargs

    def generate(self, **input_dict) -> str:
        """
        Issue a request to generate a response based on a given prompt.

        Parameters
        ----------
        input_dict : dict
            Input containing prompt data.
        """

        inputs = {self._prompt_key: [input_dict[self._prompt_key]]}

        input_dict.pop(self._prompt_key)

        return self.generate_batch(inputs=inputs, **input_dict)[0]

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

        return (await self.generate_batch_async(inputs=inputs, **input_dict, return_exceptions=False))[0]

    @typing.overload
    def generate_batch(self, inputs: dict[str, list],
                       return_exceptions: typing.Literal[True]) -> list[str | BaseException]:
        ...

    @typing.overload
    def generate_batch(self, inputs: dict[str, list], return_exceptions: typing.Literal[False]) -> list[str]:
        ...

    @typing.overload
    def generate_batch(self,
                       inputs: dict[str, list],
                       return_exceptions: bool = False) -> list[str] | list[str | BaseException]:
        ...

    def generate_batch(self,
                       inputs: dict[str, list],
                       return_exceptions=False,
                       **kwargs) -> list[str] | list[str | BaseException]:
        """
        Issue a request to generate a list of responses based on a list of prompts.

        Parameters
        ----------
        inputs : dict
            Inputs containing prompt data.
        return_exceptions : bool
            Whether to return exceptions in the output list or raise them immediately.
        **kwargs
            Additional keyword arguments for generate batch.
        """

        # Note: We dont want to use the generate_multiple implementation from nemollm because there is no retry logic.
        # As soon as one of the requests fails, the entire batch fails. Instead, we need to implement the functionality
        # listed in issue #1555 For now, we generate a warning if `return_exceptions` is True.
        if (return_exceptions):
            warnings.warn("return_exceptions==True is not currently supported by the "
                          f"{type(self).__name__}.generate_batch() method. "
                          "If an exception is raised for any item, the function will exit and raise that exception.")

        prompts = [StringPromptValue(text=p) for p in inputs[self._prompt_key]]
        final_kwargs = {**self._model_kwargs, **kwargs}

        responses = []
        try:
            generated_responses = self._client.generate_prompt(prompts=prompts, **final_kwargs)  # type: ignore
            responses = [g[0].text for g in generated_responses.generations]
        except Exception as e:
            # Leave this in here for debugging purposes
            raise e

        return responses

    @typing.overload
    async def generate_batch_async(self, inputs: dict[str, list],
                                   return_exceptions: typing.Literal[True]) -> list[str | BaseException]:
        ...

    @typing.overload
    async def generate_batch_async(self, inputs: dict[str, list],
                                   return_exceptions: typing.Literal[False]) -> list[str]:
        ...

    @typing.overload
    async def generate_batch_async(self,
                                   inputs: dict[str, list],
                                   return_exceptions: bool = False) -> list[str] | list[str | BaseException]:
        ...

    async def generate_batch_async(self,
                                   inputs: dict[str, list],
                                   return_exceptions=False,
                                   **kwargs) -> list[str] | list[str | BaseException]:
        """
        Issue an asynchronous request to generate a list of responses based on a list of prompts.

        Parameters
        ----------
        inputs : dict
            Inputs containing prompt data.
        return_exceptions : bool
            Whether to return exceptions in the output list or raise them immediately.
        **kwargs
            Additional keyword arguments for generate batch async.
        """

        # Note: We dont want to use the generate_multiple implementation from nemollm because there is no retry logic.
        # As soon as one of the requests fails, the entire batch fails. Instead, we need to implement the functionality
        # listed in issue #1555 For now, we generate a warning if `return_exceptions` is True.
        if (return_exceptions):
            warnings.warn("return_exceptions==True is not currently supported by the "
                          f"{type(self).__name__}.generate_batch_async() method. "
                          "If an exception is raised for any item, the function will exit and raise that exception.")

        prompts = [StringPromptValue(text=p) for p in inputs[self._prompt_key]]
        final_kwargs = {**self._model_kwargs, **kwargs}

        responses = []
        try:
            generated_responses = await self._client.agenerate_prompt(prompts=prompts, **final_kwargs)  # type: ignore
            responses = [g[0].text for g in generated_responses.generations]
        except Exception as e:
            # Leave this in here for debugging purposes
            raise e

        return responses


class NVFoundationLLMService(LLMService):
    """
    A service for interacting with NeMo LLM models, this class should be used to create a client for a specific model.

    Parameters
    ----------
    api_key : str, optional
        The API key for the LLM service, by default None. If `None` the API key will be read from the `NVIDIA_API_KEY`
        environment variable. If neither are present an error will be raised, by default None
    base_url : str, optional
        The api host url, by default None. If `None` the url will be read from the `NVIDIA_API_BASE` environment
        variable. If neither are present the NVIDIA default will be used, by default None
    """

    class APIKey(EnvConfigValue):
        _ENV_KEY = "NVIDIA_API_KEY"

    class BaseURL(EnvConfigValue):
        _ENV_KEY = "NVIDIA_API_BASE"
        _ALLOW_NONE = True

    def __init__(self, *, api_key: APIKey | str = None, base_url: BaseURL | str = None, **model_kwargs) -> None:
        if IMPORT_EXCEPTION is not None:
            raise ImportError(
                IMPORT_ERROR_MESSAGE.format(package='langchain-nvidia-ai-endpoints')) from IMPORT_EXCEPTION

        super().__init__()

        if not isinstance(api_key, NVFoundationLLMService.APIKey):
            api_key = NVFoundationLLMService.APIKey(api_key)

        if not isinstance(base_url, NVFoundationLLMService.BaseURL):
            base_url = NVFoundationLLMService.BaseURL(base_url)

        self._api_key = api_key
        self._base_url = base_url
        self._default_model_kwargs = model_kwargs

    def _merge_model_kwargs(self, model_kwargs: dict) -> dict:
        return {**self._default_model_kwargs, **model_kwargs}

    @property
    def api_key(self):
        return self._api_key.value

    @property
    def base_url(self):
        return self._base_url.value

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

        final_model_kwargs = self._merge_model_kwargs(model_kwargs)

        return NVFoundationLLMClient(self, model_name=model_name, **final_model_kwargs)
