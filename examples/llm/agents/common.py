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

from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.agents.agent import AgentExecutor
from langchain_openai import OpenAI

from morpheus.config import Config
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus_llm.llm import LLMEngine
from morpheus_llm.llm.nodes.extracter_node import ExtracterNode
from morpheus_llm.llm.nodes.langchain_agent_node import LangChainAgentNode
from morpheus_llm.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus_llm.stages.llm.llm_engine_stage import LLMEngineStage

logger = logging.getLogger(__name__)


def _build_agent_executor(model_name: str) -> AgentExecutor:

    llm = OpenAI(model=model_name, temperature=0.0, client=None)

    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    agent_executor = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    return agent_executor


def _build_engine(model_name: str) -> LLMEngine:

    engine = LLMEngine()

    engine.add_node("extracter", node=ExtracterNode())

    engine.add_node("agent",
                    inputs=[("/extracter")],
                    node=LangChainAgentNode(agent_executor=_build_agent_executor(model_name=model_name)))

    engine.add_task_handler(inputs=["/agent"], handler=SimpleTaskHandler())

    return engine


def build_common_pipeline(config: Config, pipe: LinearPipeline, task_payload: dict,
                          model_name: str) -> InMemorySinkStage:
    """
    Construct the elements of the pipeline common to the simple and kafka agent pipelines.
    This method should be called after the source stage has been set.
    """
    pipe.add_stage(DeserializeStage(config, task_type="llm_engine", task_payload=task_payload))

    pipe.add_stage(MonitorStage(config, description="Source rate", unit='questions'))

    pipe.add_stage(LLMEngineStage(config, engine=_build_engine(model_name=model_name)))

    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.add_stage(MonitorStage(config, description="Agent rate", unit="events", delayed_start=True))

    return sink
