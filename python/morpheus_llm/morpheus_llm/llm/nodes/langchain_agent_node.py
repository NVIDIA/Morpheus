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
import logging
import typing

from morpheus_llm.error import IMPORT_ERROR_MESSAGE
from morpheus_llm.llm import LLMContext
from morpheus_llm.llm import LLMNodeBase

logger = logging.getLogger(__name__)

IMPORT_EXCEPTION = None

try:
    from langchain_core.exceptions import OutputParserException
except ImportError as import_exc:
    IMPORT_EXCEPTION = import_exc

if typing.TYPE_CHECKING:
    from langchain.agents import AgentExecutor


class LangChainAgentNode(LLMNodeBase):
    """
    Executes a LangChain agent in an LLMEngine

    Parameters
    ----------
    agent_executor : AgentExecutor
        The agent executor to use to execute.

    replace_exceptions : bool, optional
        When `True`, replaces exceptions with `replace_exceptions_value` from the output.

    replace_exceptions_value : str, optional
        The value to replace exceptions with when `replace_exceptions` is `True`, ignored otherwise.
    """

    def __init__(self,
                 agent_executor: "AgentExecutor",
                 replace_exceptions: bool = False,
                 replace_exceptions_value: typing.Optional[str] = None):
        if IMPORT_EXCEPTION is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE.format('langchain_core')) from IMPORT_EXCEPTION

        super().__init__()

        self._agent_executor = agent_executor

        self._input_names = self._agent_executor.input_keys

        self._replace_exceptions = replace_exceptions
        self._replace_exceptions_value = replace_exceptions_value

    def get_input_names(self):
        return self._input_names

    @staticmethod
    def _is_all_lists(data: dict[str, typing.Any]) -> bool:
        return all(isinstance(v, list) for v in data.values())

    @staticmethod
    def _transform_dict_of_lists(data: dict[str, typing.Any]) -> list[dict[str, typing.Any]]:
        return [dict(zip(data, t)) for t in zip(*data.values())]

    async def _run_single(self, metadata: dict[str, typing.Any] = None, **kwargs) -> dict[str, typing.Any]:

        all_lists = self._is_all_lists(kwargs)

        # Check if all values are a list
        if all_lists:

            # Transform from dict[str, list[Any]] to list[dict[str, Any]]
            input_list = self._transform_dict_of_lists(kwargs)

            # If all metadata values are lists of the same length and the same length as the input list
            # then transform them the same way as the input list
            if (metadata is not None and self._is_all_lists(metadata)
                    and all(len(v) == len(input_list) for v in metadata.values())):
                metadata_list = self._transform_dict_of_lists(metadata)

            else:
                metadata_list = [metadata] * len(input_list)

            # Run multiple again
            results_async = [self._run_single(metadata=metadata_list[i], **x) for (i, x) in enumerate(input_list)]

            results = await asyncio.gather(*results_async, return_exceptions=True)

            # # Transform from list[dict[str, Any]] to dict[str, list[Any]]
            # results = {k: [x[k] for x in results] for k in results[0]}

            return results

        # We are not dealing with a list, so run single
        try:
            return await self._agent_executor.arun(metadata=metadata, **kwargs)
        except Exception as e:
            logger.exception("Error running agent: %s", e)
            return e

    async def execute(self, context: LLMContext) -> LLMContext:  # pylint: disable=invalid-overridden-method

        input_dict = context.get_inputs()

        results = await self._run_single(**input_dict)

        if self._replace_exceptions:
            # Processes the results to replace exceptions with a default message
            for i, answer_list in enumerate(results):
                for j, answer in enumerate(answer_list):

                    # OutputParserException is not a subclass of Exception, so we need to check for it separately
                    if isinstance(answer, (OutputParserException, Exception)):
                        # If the agent encounters a parsing error or a server error after retries, replace the error
                        # with a default value to prevent the pipeline from crashing
                        results[i][j] = self._replace_exceptions_value
                        logger.warning(
                            "Exception encountered in result[%d][%d]: %s. "
                            "Replacing with default message: '%s'.",
                            i,
                            j,
                            answer,
                            self._replace_exceptions_value)

        context.set_output(results)

        return context
