# Copyright (c) 2023, NVIDIA CORPORATION.
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
import copy
import logging
import typing

from morpheus.llm.services.llm_service import LLMClient
from morpheus.llm.services.llm_service import LLMService

logger = logging.getLogger(__name__)

IMPORT_ERROR_MESSAGE = (
    "OpenAIChatService & OpenAIChatClient require the openai package to be installed. "
    "Install it by running the following command:\n"
    "`mamba env update -n ${CONDA_DEFAULT_ENV} --file docker/conda/environments/cuda11.8_examples.yml`")

try:
    import openai
except ImportError:
    logger.error(IMPORT_ERROR_MESSAGE)


def _verify_openai():
    if 'openai' not in globals():
        raise ImportError(IMPORT_ERROR_MESSAGE)


class OpenAIChatClient(LLMClient):
    """
    Client for interacting with a specific OpenAI chat model. This class should be constructed with the
    `OpenAIChatService.get_client` method.

    Parameters
    ----------
    model_name : str
        The name of the model to interact with.

    set_assistant: bool, optional default=False
        When `True`, a second input field named `assistant` will be used to proide additional context to the model.

    model_kwargs : dict[str, typing.Any]
        Additional keyword arguments to pass to the model when generating text.
    """

    def __init__(self, model_name: str, set_assistant: bool = False, **model_kwargs: dict[str, typing.Any]) -> None:
        super().__init__()
        _verify_openai()

        self._model_name = model_name
        self._set_assistant = set_assistant
        self._prompt_key = "prompt"
        self._assistant_key = "assistant"

        # Preserve original configuration.
        self._model_kwargs = copy.deepcopy(model_kwargs)
        self._model_kwargs['temperature'] = model_kwargs.get('temperature', 0)

    def get_input_names(self) -> list[str]:
        input_names = [self._prompt_key]
        if self._set_assistant:
            input_names.append(self._assistant_key)

        return input_names

    def _create_messages(self, prompt: str, assistant: str = None) -> list[dict[str, str]]:
        messages = [
            {
                "role": "system", "content": "You are a helpful assistant."
            },
            {
                "role": "user", "content": prompt
            },
        ]

        if (self._set_assistant):
            messages.append({"role": "assistant", "content": assistant})

        return messages

    def _extract_completion(self, completion: "openai.openai_object.OpenAIObject") -> str:
        choices = completion.get('choices', [])
        if len(choices) == 0:
            raise ValueError("No choices were returned from the model.")

        content = choices[0].get('message', {}).get('content', None)
        if content is None:
            raise ValueError("No content was returned from the model.")

        return content

    def _generate(self, prompt: str, assistant: str = None) -> str:
        messages = self._create_messages(prompt, assistant)

        output = openai.ChatCompletion.create(model=self._model_name, messages=messages, **self._model_kwargs)

        return self._extract_completion(output)

    def generate(self, input_dict: dict[str, str]) -> str:
        """
        Issue a request to generate a response based on a given prompt.

        Parameters
        ----------
        input_dict : dict
            Input containing prompt data.
        """
        return self._generate(input_dict[self._prompt_key], input_dict.get(self._assistant_key))

    async def _generate_async(self, prompt: str, assistant: str = None) -> str:
        messages = self._create_messages(prompt, assistant)

        output = await openai.ChatCompletion.acreate(model=self._model_name, messages=messages, **self._model_kwargs)

        return self._extract_completion(output)

    async def generate_async(self, input_dict: dict[str, str]) -> str:
        """
        Issue an asynchronous request to generate a response based on a given prompt.

        Parameters
        ----------
        input_dict : dict
            Input containing prompt data.
        """
        return await self._generate_async(input_dict[self._prompt_key], input_dict.get(self._assistant_key))

    def generate_batch(self, inputs: dict[str, list[str]]) -> list[str]:
        """
        Issue a request to generate a list of responses based on a list of prompts.

        Parameters
        ----------
        inputs : dict
            Inputs containing prompt data.
        """
        prompts = inputs[self._prompt_key]
        assistants = None
        if (self._set_assistant):
            assistants = inputs[self._assistant_key]
            if len(prompts) != len(assistants):
                raise ValueError("The number of prompts and assistants must be equal.")

        results = []
        for (i, prompt) in enumerate(prompts):
            assistant = assistants[i] if assistants is not None else None
            results.append(self._generate(prompt, assistant))

        return results

    async def generate_batch_async(self, inputs: dict[str, list[str]]) -> list[str]:
        """
        Issue an asynchronous request to generate a list of responses based on a list of prompts.

        Parameters
        ----------
        inputs : dict
            Inputs containing prompt data.
        """
        prompts = inputs[self._prompt_key]
        assistants = None
        if (self._set_assistant):
            assistants = inputs[self._assistant_key]
            if len(prompts) != len(assistants):
                raise ValueError("The number of prompts and assistants must be equal.")

        coros = []
        for (i, prompt) in enumerate(prompts):
            assistant = assistants[i] if assistants is not None else None
            coros.append(self._generate_async(prompt, assistant))

        return await asyncio.gather(*coros)


class OpenAIChatService(LLMService):
    """
    A service for interacting with OpenAI Chat models, this class should be used to create clients.
    """

    def __init__(self) -> None:
        super().__init__()
        _verify_openai()

    def get_client(self,
                   model_name: str,
                   set_assistant: bool = False,
                   **model_kwargs: dict[str, typing.Any]) -> OpenAIChatClient:
        """
        Returns a client for interacting with a specific model. This method is the preferred way to create a client.

        Parameters
        ----------
        model_name : str
            The name of the model to create a client for.

        set_assistant: bool, optional default=False
            When `True`, a second input field named `assistant` will be used to provide additional context to the model.

        model_kwargs : dict[str, typing.Any]
            Additional keyword arguments to pass to the model when generating text.
        """

        return OpenAIChatClient(model_name=model_name, set_assistant=set_assistant, **model_kwargs)
