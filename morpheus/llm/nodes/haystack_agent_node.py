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
import typing

from morpheus.llm import LLMContext
from morpheus.llm import LLMNodeBase

IMPORT_ERROR_MESSAGE = (
    "HaystackAgentNode require the farm-haystack package to be installed. "
    "Install it by running the following command:\n"
    "`mamba install -n base -c conda-forge conda-merge`\n"
    "`conda run -n base --live-stream conda-merge docker/conda/environments/cuda${CUDA_VER}_dev.yml "
    "  docker/conda/environments/cuda${CUDA_VER}_examples.yml"
    "  > .tmp/merged.yml && mamba env update -n morpheus --file .tmp/merged.yml`")

try:
    from haystack.agents import Agent
except ImportError:
    pass

logger = logging.getLogger(__name__)


class HaystackAgentNode(LLMNodeBase):
    """
    Executes a Haystack agent in an LLMEngine

    Parameters
    ----------
    agent : Agent
        The agent to use to execute.
    return_only_answer : bool
        Return only the answer if this flag is set to True; otherwise, return the transcript, query, and answer.
        Default value is True.
    """

    def __init__(self, agent: "Agent", return_only_answer: bool = True):
        super().__init__()

        self._agent = agent
        self._return_only_answer = return_only_answer

    def get_input_names(self) -> list[str]:
        return ["query"]

    async def _run_single(self, **kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:
        all_lists = all(isinstance(v, list) for v in kwargs.values())

        # Check if all values are a list
        if all_lists:
            # Transform from dict[str, list[Any]] to list[dict[str, Any]]
            input_list = [dict(zip(kwargs, t)) for t in zip(*kwargs.values())]

            results = []

            # Run multiple queries asynchronously
            loop = asyncio.get_event_loop()

            # The asyncio loop utilizes the default executor when 'None' is passed.
            tasks = [loop.run_in_executor(None, self._agent.run, x) for x in input_list]
            results = await asyncio.gather(*tasks)

            return results

        # We are not dealing with a list, so run single
        return [self._agent.run(**kwargs)]

    def _parse_results(self, results: list[dict[str, typing.Any]]) -> list[dict[str, typing.Any]]:
        parsed_results = []

        for item in results:
            parsed_result = {}

            if self._return_only_answer:
                parsed_result["answers"] = [answer.to_dict()["answer"] for answer in item['answers']]
            else:
                parsed_result["query"] = item['query']['query']
                parsed_result["transcript"] = item['transcript']
                parsed_result["answers"] = [answer.to_dict() for answer in item['answers']]

            parsed_results.append(parsed_result)

        return parsed_results

    async def execute(self, context: LLMContext) -> LLMContext:
        input_dict = context.get_inputs()

        try:
            # Call _run_single asynchronously
            results = await self._run_single(**input_dict)
            parsed_results = self._parse_results(results)

            context.set_output(parsed_results)

        except KeyError as exe:
            logger.error("Parsing of results encountered an error: %s", exe)
        except Exception as exe:
            logger.error("Processing input encountered an error: %s", exe)

        return context
