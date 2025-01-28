# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import collections.abc
import os
import typing
from unittest import mock

import pytest

try:
    import langchain
    from langchain.agents import AgentType
    from langchain.agents import initialize_agent
    from langchain.agents import load_tools
    from langchain.agents.tools import Tool
    from langchain.utilities import serpapi
except ImportError:
    print("langchain is not installed")

import cudf

from morpheus.config import Config
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus_llm.llm import LLMEngine
from morpheus_llm.llm.nodes.extracter_node import ExtracterNode
from morpheus_llm.llm.nodes.langchain_agent_node import LangChainAgentNode
from morpheus_llm.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus_llm.stages.llm.llm_engine_stage import LLMEngineStage


def _build_agent_executor(model_name: str):

    llm = langchain.OpenAI(model=model_name, temperature=0, cache=False)

    # Explicitly construct the serpapi tool, loading it via load_tools makes it too difficult to mock
    tools = [
        Tool(
            name="Search",
            description="",
            func=serpapi.SerpAPIWrapper().run,
            coroutine=serpapi.SerpAPIWrapper().arun,
        )
    ]
    tools.extend(load_tools(["llm-math"], llm=llm))

    agent_executor = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    return agent_executor


def _build_engine(model_name: str):

    engine = LLMEngine()

    engine.add_node("extracter", node=ExtracterNode())

    engine.add_node("agent",
                    inputs=[("/extracter")],
                    node=LangChainAgentNode(agent_executor=_build_agent_executor(model_name=model_name)))

    engine.add_task_handler(inputs=["/agent"], handler=SimpleTaskHandler())

    return engine


def _run_pipeline(config: Config, source_dfs: list[cudf.DataFrame], model_name: str = "test_model"):
    completion_task = {"task_type": "completion", "task_dict": {"input_keys": ["questions"]}}

    pipe = LinearPipeline(config)

    pipe.set_source(InMemorySourceStage(config, dataframes=source_dfs))

    pipe.add_stage(DeserializeStage(config, task_type="llm_engine", task_payload=completion_task))

    pipe.add_stage(LLMEngineStage(config, engine=_build_engine(model_name=model_name)))

    pipe.add_stage(InMemorySinkStage(config))

    pipe.run()


@pytest.mark.usefixtures("openai", "restore_environ")
@pytest.mark.cpu_mode
@pytest.mark.benchmark
@mock.patch("langchain.utilities.serpapi.SerpAPIWrapper.aresults")
@mock.patch("langchain.OpenAI._agenerate", autospec=True)  # autospec is needed as langchain will inspect the function
def test_agents_simple_pipe(mock_openai_agenerate: mock.AsyncMock,
                            mock_serpapi_aresults: mock.AsyncMock,
                            mock_openai_request_time: float,
                            mock_serpapi_request_time: float,
                            benchmark: collections.abc.Callable[[collections.abc.Callable], typing.Any],
                            config: Config):
    os.environ.update({'OPENAI_API_KEY': 'test_api_key', 'SERPAPI_API_KEY': 'test_api_key'})

    from langchain.schema import Generation
    from langchain.schema import LLMResult

    assert serpapi.SerpAPIWrapper().aresults is mock_serpapi_aresults

    model_name = "test_model"

    mock_responses = [
        LLMResult(generations=[[
            Generation(text="I should use a search engine to find information about unittests.\n"
                       "Action: Search\nAction Input: \"unittests\"",
                       generation_info={
                           'finish_reason': 'stop', 'logprobs': None
                       })
        ]],
                  llm_output={
                      'token_usage': {}, 'model_name': model_name
                  }),
        LLMResult(generations=[[
            Generation(text="I now know the final answer.\nFinal Answer: 3.99.",
                       generation_info={
                           'finish_reason': 'stop', 'logprobs': None
                       })
        ]],
                  llm_output={
                      'token_usage': {}, 'model_name': model_name
                  })
    ]

    async def _mock_openai_agenerate(self, *args, **kwargs):  # pylint: disable=unused-argument
        nonlocal mock_responses
        call_count = getattr(self, '_unittest_call_count', 0)
        response = mock_responses[call_count % 2]

        # The OpenAI object will raise a ValueError if we attempt to set the attribute directly or use setattr
        self.__dict__['_unittest_call_count'] = call_count + 1
        await asyncio.sleep(mock_openai_request_time)
        return response

    mock_openai_agenerate.side_effect = _mock_openai_agenerate

    async def _mock_serpapi_aresults(*args, **kwargs):  # pylint: disable=unused-argument
        await asyncio.sleep(mock_serpapi_request_time)
        return {
            'answer_box': {
                'answer': '25 years', 'link': 'http://unit.test', 'people_also_search_for': []
            },
            'inline_people_also_search_for': [],
            'knowledge_graph': {},
            'organic_results': [],
            'pagination': {},
            'related_questions': [],
            'related_searches': [],
            'search_information': {},
            'search_metadata': {},
            'search_parameters': {},
            'serpapi_pagination': None
        }

    mock_serpapi_aresults.side_effect = _mock_serpapi_aresults

    source_df = cudf.DataFrame(
        {"questions": ["Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"]})

    benchmark(_run_pipeline, config, source_dfs=[source_df], model_name=model_name)
