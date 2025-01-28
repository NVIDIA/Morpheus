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

import os
import re
from unittest import mock

import pytest

import cudf

from _utils import assert_results
from morpheus.config import Config
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.concat_df import concat_dataframes
from morpheus_llm.llm import LLMEngine
from morpheus_llm.llm.nodes.extracter_node import ExtracterNode
from morpheus_llm.llm.nodes.langchain_agent_node import LangChainAgentNode
from morpheus_llm.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus_llm.stages.llm.llm_engine_stage import LLMEngineStage

try:
    from langchain.agents import AgentType
    from langchain.agents import initialize_agent
    from langchain.agents import load_tools
    from langchain.agents.tools import Tool
    from langchain.schema import Generation
    from langchain.schema import LLMResult
    from langchain_community.utilities import serpapi
    from langchain_openai import OpenAI  # pylint: disable=no-name-in-module
except ImportError:
    pass


@pytest.fixture(name="questions")
def questions_fixture():
    return ["Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"]


def _build_agent_executor(model_name: str):
    llm = OpenAI(model=model_name, temperature=0, cache=False)

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


def _run_pipeline(config: Config,
                  questions: list[str],
                  model_name: str = "test_model",
                  expected_df: cudf.DataFrame = None) -> InMemorySinkStage:
    source_df = cudf.DataFrame({"questions": questions})

    completion_task = {"task_type": "completion", "task_dict": {"input_keys": ["questions"]}}

    pipe = LinearPipeline(config)

    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))

    pipe.add_stage(DeserializeStage(config, task_type="llm_engine", task_payload=completion_task))

    pipe.add_stage(LLMEngineStage(config, engine=_build_engine(model_name=model_name)))

    if expected_df is not None:
        sink = pipe.add_stage(CompareDataFrameStage(config, compare_df=expected_df))
    else:
        sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    return sink


@pytest.mark.usefixtures("openai", "openai_api_key", "serpapi_api_key")
def test_agents_simple_pipe_integration_openai(config: Config, questions: list[str]):
    sink = _run_pipeline(config, questions=questions, model_name="gpt-3.5-turbo-instruct")

    result_df = concat_dataframes(sink.get_messages())
    assert len(result_df.columns) == 2
    assert sorted(result_df.columns) == ["questions", "response"]

    response_txt = result_df.response.iloc[0]
    response_match = re.match(r".*(\d+\.\d+)\.?$", response_txt)
    assert response_match is not None
    assert float(response_match.group(1)) >= 3.7


@pytest.mark.usefixtures("langchain_community", "langchain_openai", "numexpr", "openai", "restore_environ")
@mock.patch("langchain_community.utilities.serpapi.SerpAPIWrapper.aresults")
@mock.patch("langchain_openai.OpenAI._agenerate",
            autospec=True)  # autospec is needed as langchain will inspect the function
def test_agents_simple_pipe(mock_openai_agenerate: mock.AsyncMock,
                            mock_serpapi_aresults: mock.AsyncMock,
                            config: Config,
                            questions: list[str]):
    os.environ.update({'OPENAI_API_KEY': 'test_api_key', 'SERPAPI_API_KEY': 'test_api_key'})

    assert serpapi.SerpAPIWrapper().aresults is mock_serpapi_aresults

    model_name = "test_model"

    mock_openai_agenerate.side_effect = [
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

    mock_serpapi_aresults.return_value = {
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

    expected_df = cudf.DataFrame({'questions': questions, 'response': ["3.99."]})

    sink = _run_pipeline(config, questions=questions, model_name=model_name, expected_df=expected_df)

    assert len(mock_openai_agenerate.mock_calls) == 2
    mock_serpapi_aresults.assert_awaited_once()

    assert_results(sink.get_results())
