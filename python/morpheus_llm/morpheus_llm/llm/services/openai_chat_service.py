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
import asyncio
import copy
import logging
import os
import time
import typing
from contextlib import contextmanager
from textwrap import dedent

import appdirs

from morpheus.utils.env_config_value import EnvConfigValue
from morpheus_llm.error import IMPORT_ERROR_MESSAGE
from morpheus_llm.llm.services.llm_service import LLMClient
from morpheus_llm.llm.services.llm_service import LLMService

logger = logging.getLogger(__name__)

IMPORT_EXCEPTION = None

try:
    import openai
    import openai.types.chat
    import openai.types.chat.chat_completion
    from openai.types.chat.completion_create_params import ResponseFormat

except ImportError as import_exc:
    IMPORT_EXCEPTION = import_exc


class _ApiLogger:
    """
    Simple class that allows passing back and forth the inputs and outputs of an API call via a context manager.
    """

    log_template: typing.ClassVar[str] = dedent("""
        ============= MESSAGE %d START ==============
                        --- Input ---
        %s
                        --- Output --- (%f ms)
        %s
        =============  MESSAGE %d END ==============
        """).strip("\n")

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
    parent : OpenAIChatService
        The parent service for this client.

    model_name : str
        The name of the model to interact with.

    set_assistant: bool, optional
        When `True`, a second input field named `assistant` will be used to proide additional context to the model, by
        default False

    max_retries: int, optional
        The maximum number of retries to attempt when making a request to the OpenAI API, by default 10

    json: bool, optional
        When `True`, the response will be returned as a JSON object, by default False

    model_kwargs : dict[str, typing.Any]
        Additional keyword arguments to pass to the model when generating text.
    """

    _prompt_key: str = "prompt"
    _assistant_key: str = "assistant"

    def __init__(self,
                 parent: "OpenAIChatService",
                 *,
                 model_name: str,
                 set_assistant: bool = False,
                 max_retries: int = 10,
                 json=False,
                 **model_kwargs) -> None:
        if IMPORT_EXCEPTION is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE.format(package='openai')) from IMPORT_EXCEPTION

        super().__init__()

        assert parent is not None, "Parent service cannot be None."

        self._parent = parent

        self._model_name = model_name
        self._set_assistant = set_assistant
        self._prompt_key = "prompt"
        self._assistant_key = "assistant"
        self._json = json

        # Preserve original configuration.
        self._model_kwargs = copy.deepcopy(model_kwargs)

        if (self._json):
            self._model_kwargs["response_format"] = ResponseFormat(type="json_object")

        # Create the client objects for both sync and async
        self._client = openai.OpenAI(max_retries=max_retries,
                                     organization=self._parent._org_id.value,
                                     api_key=self._parent._api_key.value,
                                     base_url=self._parent._base_url.value)
        self._client_async = openai.AsyncOpenAI(max_retries=max_retries,
                                                organization=self._parent._org_id.value,
                                                api_key=self._parent._api_key.value,
                                                base_url=self._parent._base_url.value)

    @property
    def model_name(self):
        """
        Get the name of the model associated with this client.

        Returns
        -------
        str
            The name of the model.
        """
        return self._model_name

    @property
    def model_kwargs(self):
        """
        Get the keyword args that will be passed to the model when calling generation functions.

        Returns
        -------
        dict
            The keyword arguments dictionary.
        """
        # Return a copy to avoid modification of the original
        return self._model_kwargs.copy()

    def get_input_names(self) -> list[str]:
        input_names = [self._prompt_key]
        if self._set_assistant:
            input_names.append(self._assistant_key)

        return input_names

    @contextmanager
    def _api_logger(self, inputs: typing.Any):

        message_id = self._parent._get_message_id()
        start_time = time.time()

        api_logger = _ApiLogger(message_id=message_id, inputs=inputs)

        yield api_logger

        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000.0

        self._parent._logger.info(_ApiLogger.log_template,
                                  message_id,
                                  api_logger.inputs,
                                  duration_ms,
                                  api_logger.outputs,
                                  message_id)

    def _create_messages(self,
                         prompt: str,
                         assistant: str = None) -> list["openai.types.chat.ChatCompletionMessageParam"]:
        messages: list[openai.types.chat.ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]

        if (self._set_assistant and assistant is not None):
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

        output: openai.types.chat.chat_completion.ChatCompletion = self._client.chat.completions.create(
            model=self._model_name, messages=messages, **self._model_kwargs)

        return self._extract_completion(output)

    def generate(self, **input_dict) -> str:
        """
        Issue a request to generate a response based on a given prompt.

        Parameters
        ----------
        input_dict : dict
            Input containing prompt data.
        """

        prompt: str = input_dict[self._prompt_key]
        assistant: str = input_dict.get(self._assistant_key, None)

        return self._generate(prompt=prompt, assistant=assistant)

    async def _generate_async(self, prompt: str, assistant: str = None) -> str:

        messages = self._create_messages(prompt, assistant)

        with self._api_logger(inputs=messages) as msg_logger:

            try:
                output = await self._client_async.chat.completions.create(model=self._model_name,
                                                                          messages=messages,
                                                                          **self._model_kwargs)
            except Exception as exc:
                self._parent._logger.error("Error generating completion: %s", exc)
                raise

            msg_logger.set_output(output)

        return self._extract_completion(output)

    async def generate_async(self, **input_dict) -> str:
        """
        Issue an asynchronous request to generate a response based on a given prompt.

        Parameters
        ----------
        input_dict : dict
            Input containing prompt data.
        """
        return await self._generate_async(input_dict[self._prompt_key], input_dict.get(self._assistant_key, None))

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

    def generate_batch(self, inputs: dict[str, list], return_exceptions=False) -> list[str] | list[str | BaseException]:
        """
        Issue a request to generate a list of responses based on a list of prompts.

        Parameters
        ----------
        inputs : dict
            Inputs containing prompt data.
        return_exceptions : bool
            Whether to return exceptions in the output list or raise them immediately.
        """
        prompts = inputs[self._prompt_key]
        assistants = None
        if (self._set_assistant):
            assistants = inputs[self._assistant_key]
            if len(prompts) != len(assistants):
                raise ValueError("The number of prompts and assistants must be equal.")

        results = []
        for (i, prompt) in enumerate(prompts):
            p: str = prompt
            assistant: str | None = assistants[i] if assistants is not None else None

            try:
                results.append(self._generate(p, assistant))
            except BaseException as e:

                if return_exceptions:
                    results.append(e)
                else:
                    raise

        return results

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
                                   return_exceptions=False) -> list[str] | list[str | BaseException]:
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
        assistants = None
        if (self._set_assistant):
            assistants = inputs[self._assistant_key]
            if len(prompts) != len(assistants):
                raise ValueError("The number of prompts and assistants must be equal.")

        coros = []
        for (i, prompt) in enumerate(prompts):
            assistant = assistants[i] if assistants is not None else None
            coros.append(self._generate_async(prompt, assistant))

        return await asyncio.gather(*coros, return_exceptions=return_exceptions)


class OpenAIChatService(LLMService):
    """
    A service for interacting with OpenAI Chat models, this class should be used to create clients.

    Parameters
    ----------
    api_key : str, optional
        The API key for the LLM service, by default None. If `None` the API key will be read from the
        `OPENAI_API_KEY` environment variable. If neither are present an error will be raised.
    org_id : str, optional
        The organization ID for the LLM service, by default None. If `None` the organization ID will be read from
        the `OPENAI_ORG_ID` environment variable. This value is only required if the account associated with the
        `api_key` is a member of multiple organizations, by default None
    base_url : str, optional
        The api host url, by default None. If `None` the url will be read from the `OPENAI_BASE_URL` environment
        variable. If neither are present the OpenAI default will be used, by default None
    default_model_kwargs : dict, optional
        Default arguments to use when creating a client via the `get_client` function. Any argument specified here
        will automatically be used when calling `get_client`. Arguments specified in the `get_client` function will
        overwrite default values specified here. This is useful to set model arguments before creating multiple
        clients. By default None

    """

    class APIKey(EnvConfigValue):
        _ENV_KEY = "OPENAI_API_KEY"

    class OrgId(EnvConfigValue):
        _ENV_KEY = "OPENAI_ORG_ID"
        _ALLOW_NONE = True

    class BaseURL(EnvConfigValue):
        _ENV_KEY = "OPENAI_BASE_URL"
        _ALLOW_NONE = True

    def __init__(self,
                 *,
                 api_key: APIKey | str = None,
                 org_id: OrgId | str = None,
                 base_url: BaseURL | str = None,
                 default_model_kwargs: dict = None) -> None:

        if IMPORT_EXCEPTION is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE.format(package='openai')) from IMPORT_EXCEPTION

        super().__init__()

        if not isinstance(api_key, OpenAIChatService.APIKey):
            api_key = OpenAIChatService.APIKey(api_key)

        if not isinstance(org_id, OpenAIChatService.OrgId):
            org_id = OpenAIChatService.OrgId(org_id)

        if not isinstance(base_url, OpenAIChatService.BaseURL):
            base_url = OpenAIChatService.BaseURL(base_url)

        self._api_key = api_key
        self._org_id = org_id
        self._base_url = base_url
        self._default_model_kwargs = default_model_kwargs or {}

        self._logger = logging.getLogger(f"{__package__}.{OpenAIChatService.__name__}")

        # Dont propagate up to the default logger. Just log to file
        self._logger.propagate = False

        log_file = os.path.join(appdirs.user_log_dir(appauthor="NVIDIA", appname="morpheus"), "openai.log")

        # Ensure the log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

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
                   *,
                   model_name: str,
                   set_assistant: bool = False,
                   max_retries: int = 10,
                   json=False,
                   **model_kwargs) -> OpenAIChatClient:
        """
        Returns a client for interacting with a specific model. This method is the preferred way to create a client.

        Parameters
        ----------
        model_name : str
            The name of the model to create a client for.

        set_assistant: bool, optional
            When `True`, a second input field named `assistant` will be used to proide additional context to the model,
            by default False

        max_retries: int, optional
            The maximum number of retries to attempt when making a request to the OpenAI API, by default 10

        json: bool, optional
            When `True`, the response will be returned as a JSON object, by default False

        model_kwargs : dict[str, typing.Any]
            Additional keyword arguments to pass to the model when generating text. Arguments specified here will
            overwrite the `default_model_kwargs` set in the service constructor
        """

        final_model_kwargs = {**self._default_model_kwargs, **model_kwargs}

        return OpenAIChatClient(self,
                                model_name=model_name,
                                set_assistant=set_assistant,
                                max_retries=max_retries,
                                json=json,
                                **final_model_kwargs)
