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
import warnings

from morpheus.llm.services.llm_service import LLMClient
from morpheus.llm.services.llm_service import LLMService

logger = logging.getLogger(__name__)

IMPORT_EXCEPTION = None
IMPORT_ERROR_MESSAGE = (
    "NemoLLM not found. Install it and other additional dependencies by running the following command:\n"
    "`conda env update --solver=libmamba -n morpheus "
    "--file conda/environments/dev_cuda-121_arch-x86_64.yaml --prune`")

try:
    import nemollm
except ImportError as import_exc:
    IMPORT_EXCEPTION = import_exc


class NeMoLLMClient(LLMClient):
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

    def __init__(self, parent: "NeMoLLMService", *, model_name: str, **model_kwargs) -> None:
        if IMPORT_EXCEPTION is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE) from IMPORT_EXCEPTION

        super().__init__()

        assert parent is not None, "Parent service cannot be None."

        self._parent = parent
        self._model_name = model_name
        self._model_kwargs = model_kwargs
        self._prompt_key = "prompt"

    def get_input_names(self) -> list[str]:
        return [self._prompt_key]

    def generate(self, **input_dict) -> str:
        """
        Issue a request to generate a response based on a given prompt.

        Parameters
        ----------
        input_dict : dict
            Input containing prompt data.
        """
        return self.generate_batch({self._prompt_key: [input_dict[self._prompt_key]]}, return_exceptions=False)[0]

    async def generate_async(self, **input_dict) -> str:
        """
        Issue an asynchronous request to generate a response based on a given prompt.

        Parameters
        ----------
        input_dict : dict
            Input containing prompt data.
        """
        return (await self.generate_batch_async({self._prompt_key: [input_dict[self._prompt_key]]},
                                                return_exceptions=False))[0]

    @typing.overload
    def generate_batch(self,
                       inputs: dict[str, list],
                       return_exceptions: typing.Literal[True] = True, **kwargs) -> list[str | BaseException]:
        ...

    @typing.overload
    def generate_batch(self, inputs: dict[str, list], return_exceptions: typing.Literal[False] = False, **kwargs) -> list[str]:
        ...

    def generate_batch(self, inputs: dict[str, list], return_exceptions=False, **kwargs) -> list[str] | list[str | BaseException]:
        """
        Issue a request to generate a list of responses based on a list of prompts.

        Parameters
        ----------
        inputs : dict
            Inputs containing prompt data.
        return_exceptions : bool
            Whether to return exceptions in the output list or raise them immediately.
        """

        # Note: We dont want to use the generate_multiple implementation from nemollm because there is no retry logic.
        # As soon as one of the requests fails, the entire batch fails. Instead, we need to implement the functionality
        # listed in issue #1555 For now, we generate a warning if `return_exceptions` is True.
        if (return_exceptions):
            warnings.warn("return_exceptions==True is not currently supported by the NeMoLLMClient. "
                          "If an exception is raised for any item, the function will exit and raise that exception.")

        return typing.cast(
            list[str],
            self._parent._conn.generate_multiple(model=self._model_name,
                                                 prompts=inputs[self._prompt_key],
                                                 return_type="text",
                                                 **self._model_kwargs))

    async def _process_one_async(self, prompt: str) -> str:
        iterations = 0
        errors = []

        while iterations < self._parent._retry_count:
            fut = await asyncio.wrap_future(
                self._parent._conn.generate(model=self._model_name,
                                            prompt=prompt,
                                            return_type="async",
                                            **self._model_kwargs))  # type: ignore

            result: dict = nemollm.NemoLLM.post_process_generate_response(
                fut, return_text_completion_only=False)  # type: ignore

            if result.get('status', None) == 'fail':
                iterations += 1
                errors.append(result.get('msg', 'Unknown error'))
                continue

            return result['text']

        raise RuntimeError(
            f"Failed to generate response for prompt '{prompt}' after {self._parent._retry_count} attempts. "
            f"Errors: {errors}")

    @typing.overload
    async def generate_batch_async(self,
                                   inputs: dict[str, list],
                                   return_exceptions: typing.Literal[True] = True, **kwargs) -> list[str | BaseException]:
        ...

    @typing.overload
    async def generate_batch_async(self,
                                   inputs: dict[str, list],
                                   return_exceptions: typing.Literal[False] = False, **kwargs) -> list[str]:
        ...

    async def generate_batch_async(self,
                                   inputs: dict[str, list],
                                   return_exceptions=False, **kwargs) -> list[str] | list[str | BaseException]:
        """
        Issue an asynchronous request to generate a list of responses based on a list of prompts.

        Parameters
        ----------
        inputs : dict
            Inputs containing prompt data.
        return_exceptions : bool
            Whether to return exceptions in the output list or raise them immediately.
        """
        prompts = inputs[self._prompt_key]

        futures = [self._process_one_async(p) for p in prompts]

        results = await asyncio.gather(*futures, return_exceptions=return_exceptions)

        return results


class NeMoLLMService(LLMService):
    """
    A service for interacting with NeMo LLM models, this class should be used to create a client for a specific model.
    """

    def __init__(self, *, api_key: str = None, org_id: str = None, retry_count=5) -> None:
        """
        Creates a service for interacting with NeMo LLM models.

        Parameters
        ----------
        api_key : str, optional
            The API key for the LLM service, by default None. If `None` the API key will be read from the `NGC_API_KEY`
            environment variable. If neither are present an error will be raised., by default None
        org_id : str, optional
            The organization ID for the LLM service, by default None. If `None` the organization ID will be read from
            the `NGC_ORG_ID` environment variable. This value is only required if the account associated with the
            `api_key` is a member of multiple NGC organizations., by default None
        retry_count : int, optional
            The number of times to retry a request before raising an exception, by default 5

        """

        if IMPORT_EXCEPTION is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE) from IMPORT_EXCEPTION

        super().__init__()
        api_key = api_key if api_key is not None else os.environ.get("NGC_API_KEY", None)
        org_id = org_id if org_id is not None else os.environ.get("NGC_ORG_ID", None)

        self._retry_count = retry_count

        self._conn = nemollm.NemoLLM(
            api_host=os.environ.get("NGC_API_BASE", None),
            # The client must configure the authentication and authorization parameters
            # in accordance with the API server security policy.
            # Configure Bearer authorization
            api_key=api_key,

            # If you are in more than one LLM-enabled organization, you must
            # specify your org ID in the form of a header. This is optional
            # if you are only in one LLM-enabled org.
            org_id=org_id,
        )

    def get_client(self, *, model_name: str, **model_kwargs) -> NeMoLLMClient:
        """
        Returns a client for interacting with a specific model. This method is the preferred way to create a client.

        Parameters
        ----------
        model_name : str
            The name of the model to create a client for.

        model_kwargs : dict[str, typing.Any]
            Additional keyword arguments to pass to the model when generating text.
        """

        return NeMoLLMClient(self, model_name=model_name, **model_kwargs)
