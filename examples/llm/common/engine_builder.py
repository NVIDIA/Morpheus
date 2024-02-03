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

import os

from haystack.agents import Agent
from haystack.agents import Tool
from haystack.agents.base import ToolsManager
from haystack.nodes import PromptNode
from haystack.nodes import PromptTemplate
from haystack.nodes.retriever.web import WebRetriever
from haystack.pipelines import WebQAPipeline
from langchain import OpenAI  # pylint: disable=no-name-in-module
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.agents.agent import AgentExecutor
from llama_index import VectorStoreIndex
from llama_index.langchain_helpers.agents import IndexToolConfig
from llama_index.langchain_helpers.agents import LlamaToolkit
from llama_index.vector_stores import MilvusVectorStore

from morpheus.llm import LLMEngine
from morpheus.llm.nodes.extracter_node import ExtracterNode
from morpheus.llm.nodes.haystack_agent_node import HaystackAgentNode
from morpheus.llm.nodes.langchain_agent_node import LangChainAgentNode
from morpheus.llm.task_handlers.simple_task_handler import SimpleTaskHandler


def build_engine_with_agent_node(model_name: str, llm_orch: str) -> LLMEngine:
    """
    Constructs an LLMEngine with an agent node based on the provided LLM orchestration framework name.

    Parameters
    ----------
    model_name : str
        LLM Model name.
    llm_orch : str
        LLM orchestration framework name, that is used to create an agent.

    Returns
    -------
    LLMEngine
        LLMEngine with agent node.
    """

    engine = LLMEngine()

    engine.add_node("extracter", node=ExtracterNode())

    agent_node = None

    if llm_orch == "langchain":
        agent_node = LangChainAgentNode(agent_executor=_build_langchain_agent_executor(model_name=model_name))
    elif llm_orch == "haystack":
        agent_node = HaystackAgentNode(agent=_build_haystack_agent(model_name=model_name))
    elif llm_orch == "llama_index":
        agent_node = LangChainAgentNode(agent_executor=_build_llama_index_agent_executor(model_name=model_name))
    else:
        raise RuntimeError(f"LLM orchestration framework '{llm_orch}' is not supported yet.")

    engine.add_node("agent", inputs=[("/extracter")], node=agent_node)

    engine.add_task_handler(inputs=["/agent"], handler=SimpleTaskHandler())

    return engine


def _build_langchain_agent_executor(model_name: str) -> AgentExecutor:
    """
    Builds a LangChain agent executor.

    Parameters
    ----------
    model_name : str
        LLM Model name.

    Returns
    -------
    AgentExecutor
        Langchain AgentExecutor instance.
    """

    llm = OpenAI(model=model_name, temperature=0)

    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    agent_executor = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    return agent_executor


def _build_llama_index_agent_executor(model_name: str, ) -> AgentExecutor:

    llm = OpenAI(model=model_name, temperature=0)

    vector_store = MilvusVectorStore(dim=1536, overwrite=False)
    index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = index.as_query_engine()

    query_enige_config = IndexToolConfig(
        query_engine=query_engine,
        name="Brief biography of celebrities",
        description="Category of the celebrity, one of [Sports, Entertainment, Business, Music]",
        tool_kwargs={"return_direct": False})
    toolkit = LlamaToolkit(index_configs=[query_enige_config])
    langchain_tools = load_tools(["llm-math"], llm=llm)
    tools = (toolkit.get_tools() + langchain_tools)

    # The 'initialize_agent' function in langchain is being utilized by llamaindex's 'create_llama_agent'
    # behind the scenes.
    agent_executor = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    return agent_executor


def _build_haystack_agent(model_name: str) -> Agent:
    """
    Builds a Haystack agent.

    Parameters
    ----------
    model_name : str
        LLM Model name.

    Returns
    -------
    Agent
        Haystack Agent instance.
    """

    search_key = os.environ.get("SERPER_API_KEY")
    if not search_key:
        raise ValueError("Ensure to configure the SERPER_API_KEY environment variable.")

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("Ensure to configure the OPENAI_API_KEY environment variable.")

    web_prompt_node = PromptNode(
        model_name,
        api_key=openai_key,
        max_length=256,
        default_prompt_template="deepset/question-answering",
    )

    calc_prompt_node = PromptNode(model_name,
                                  api_key=openai_key,
                                  default_prompt_template="""
        Calculate the result of the following mathematical expression:

        Expression: ({query})
        """)

    web_retriever = WebRetriever(api_key=search_key)
    web_qa_pipeline = WebQAPipeline(retriever=web_retriever, prompt_node=web_prompt_node)

    prompt_template = PromptTemplate("deepset/zero-shot-react")
    prompt_node = PromptNode(model_name,
                             api_key=os.environ.get("OPENAI_API_KEY"),
                             max_length=512,
                             stop_words=["Observation:"])

    web_qa_tool = Tool(
        name="Search",
        pipeline_or_node=web_qa_pipeline,
        description="Useful when you need to search for answers online.",
        output_variable="results",
    )

    calc_tool = Tool(name="Calculator",
                     pipeline_or_node=calc_prompt_node,
                     description="Useful when you need to math calculations.")

    agent = Agent(prompt_node=prompt_node,
                  prompt_template=prompt_template,
                  tools_manager=ToolsManager([web_qa_tool, calc_tool]))

    return agent
