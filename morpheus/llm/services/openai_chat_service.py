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
import copy
import json
import logging
import os
import time
import typing
from contextlib import contextmanager
from textwrap import dedent

import appdirs
import httpx
import tenacity
from langchain_core.caches import RETURN_VAL_TYPE
from langchain_core.caches import BaseCache
from openai.types.chat import ChatCompletion
from pydantic import Field

from morpheus.llm.services.llm_service import LLMClient
from morpheus.llm.services.llm_service import LLMService
from morpheus.utils.http_utils import retry_async

logger = logging.getLogger(__name__)

IMPORT_EXCEPTION = None
IMPORT_ERROR_MESSAGE = ("OpenAIChatService & OpenAIChatClient require the openai package to be installed. "
                        "Install it by running the following command:\n"
                        "`conda env update --solver=libmamba -n morpheus "
                        "--file morpheus/conda/environments/dev_cuda-121_arch-x86_64.yaml --prune`")

try:
    import openai
    import openai.types.chat.chat_completion
except ImportError as import_exc:
    IMPORT_EXCEPTION = import_exc


class ApiLogger:

    def __init__(self, *, message_id: int, inputs: typing.Any) -> None:

        self.message_id = message_id
        self.inputs = inputs
        self.outputs = None

    def set_output(self, output: typing.Any) -> None:
        self.outputs = output


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

    class Config:
        underscore_attrs_are_private = True

    model_name: str
    set_assistant: bool = False
    model_kwargs: dict[str, typing.Any]

    _parent: "OpenAIChatService"

    _prompt_key: str = "prompt"
    _assistant_key: str = "assistant"

    _client: openai.OpenAI
    _client_async: openai.AsyncOpenAI

    def __init__(self,
                 parent: "OpenAIChatService",
                 *,
                 model_name: str,
                 set_assistant: bool = False,
                 **model_kwargs: dict[str, typing.Any]) -> None:
        if IMPORT_EXCEPTION is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE) from IMPORT_EXCEPTION

        super().__init__(model_name=model_name, set_assistant=set_assistant, model_kwargs=copy.deepcopy(model_kwargs))

        self._parent = parent

        # self._prompt_key = "prompt"
        # self._assistant_key = "assistant"

        # Preserve original configuration.
        # self._model_kwargs = copy.deepcopy(model_kwargs)
        # self._model_kwargs['temperature'] = model_kwargs.get('temperature', 0)

        self._client = openai.OpenAI(http_client=self._parent._http_client, )
        self._client_async = openai.AsyncOpenAI(http_client=self._parent._http_client_async)

    def get_input_names(self) -> list[str]:
        input_names = [self._prompt_key]
        if self.set_assistant:
            input_names.append(self._assistant_key)

        return input_names

    @contextmanager
    def _api_logger(self, inputs: typing.Any):

        message_id = self._parent._get_message_id()
        start_time = time.time()

        api_logger = ApiLogger(message_id=message_id, inputs=inputs)

        yield api_logger

        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000.0

        log_str = dedent("""
        ============= MESSAGE %d START ==============
                        --- Input ---
        %s
                        --- Output --- (%f ms)
        %s
        =============  MESSAGE %d END ==============
        """).strip("\n")

        self._parent._logger.info(log_str, message_id, api_logger.inputs, duration_ms, api_logger.outputs, message_id)

    def _create_messages(self, prompt: str, assistant: str = None) -> list[dict[str, str]]:
        messages = [
            {
                "role": "user", "content": prompt
            },
        ]

        if (self.set_assistant and assistant is not None):
            messages.append({"role": "assistant", "content": assistant})

        return messages

    def _extract_completion(self, completion: "openai.types.chat.chat_completion.ChatCompletion") -> str:
        choices = completion.choices
        if len(choices) == 0:
            raise ValueError("No choices were returned from the model.")

        content = choices[0].message.content
        if content is None:
            raise ValueError("No content was returned from the model.")

        return content

    def _generate(self, prompt: str, assistant: str = None) -> str:
        messages = self._create_messages(prompt, assistant)

        output = self._client.chat.completions.create(model=self.model_name, messages=messages, **self.model_kwargs)

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

    @retry_async()
    async def _client_create(self, model: str, messages: list[dict[str, str]],
                             **kwargs: dict[str, typing.Any]) -> ChatCompletion:

        def test():
            start_time = time.time()

            resp = self._client.chat.completions.create(model=model, messages=messages, **kwargs)

            stop_time = time.time()

            duration_ms = (stop_time - start_time) * 1000.0

            print("Duration: ", duration_ms)

            return resp

        test()
        test()
        test()
        test()

        return test()

    async def _generate_async(self, prompt: str, assistant: str = None) -> str:

        messages = self._create_messages(prompt, assistant)

        with self._api_logger(inputs=messages) as msg_logger:

            if (self._parent._cache is not None):
                llm_str = self.json()
                prompt_str = json.dumps(messages)

                cached_output = self._parent._cache.lookup(prompt_str, llm_str)

                if (isinstance(cached_output, list)):
                    output = ChatCompletion(choices=cached_output)
                else:
                    output = await self._client_create(model=self.model_name, messages=messages, **self.model_kwargs)

                    self._parent._cache.update(prompt_str, llm_str, output.choices)
            else:

                try:
                    output = await self._client_create(model=self.model_name, messages=messages, **self.model_kwargs)
                except Exception as exc:
                    self._parent._logger.error("Error generating completion: %s", exc)

            msg_logger.set_output(output)

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
        if (self.set_assistant):
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
        if (self.set_assistant):
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

    def __init__(self,
                 *,
                 http_client: httpx.Client = None,
                 http_client_async: httpx.AsyncClient = None,
                 cache: BaseCache = None,
                 default_model_kwargs: dict = None) -> None:
        if IMPORT_EXCEPTION is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE) from IMPORT_EXCEPTION

        super().__init__()

        self._http_client = http_client
        self._http_client_async = http_client_async
        self._cache = cache
        self._default_model_kwargs = default_model_kwargs or {}

        self._logger = logging.getLogger(f"{__package__}.{OpenAIChatService.__name__}")

        # Dont propagate up to the default logger. Just log to file
        self._logger.propagate = False

        log_file = os.path.join(appdirs.user_log_dir(appauthor="NVIDIA", appname="morpheus"), "openai.log")

        # Add a file handler
        file_handler = logging.FileHandler(log_file)

        self._logger.addHandler(file_handler)
        self._logger.setLevel(logging.INFO)

        self._logger.info("OpenAI Chat Service started.")

        self._message_count = 0

    def _get_message_id(self):

        self._message_count += 1

        return self._message_count

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
            When `True`, a second input field named `assistant` will be used to proide additional context to the model.

        model_kwargs : dict[str, typing.Any]
            Additional keyword arguments to pass to the model when generating text.
        """

        final_model_kwargs = {**self._default_model_kwargs, **model_kwargs}

        return OpenAIChatClient(self, model_name=model_name, set_assistant=set_assistant, **final_model_kwargs)
