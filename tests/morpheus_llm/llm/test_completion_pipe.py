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

import logging
from unittest import mock

import pytest

import cudf

from _utils import assert_results
from _utils.environment import set_env
from _utils.llm import mk_mock_openai_response
from morpheus.config import Config
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus_llm.llm import LLMEngine
from morpheus_llm.llm.nodes.extracter_node import ExtracterNode
from morpheus_llm.llm.nodes.llm_generate_node import LLMGenerateNode
from morpheus_llm.llm.nodes.prompt_template_node import PromptTemplateNode
from morpheus_llm.llm.services.llm_service import LLMClient
from morpheus_llm.llm.services.nemo_llm_service import NeMoLLMService
from morpheus_llm.llm.services.openai_chat_service import OpenAIChatService
from morpheus_llm.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus_llm.stages.llm.llm_engine_stage import LLMEngineStage

logger = logging.getLogger(__name__)


def _build_engine(llm_client: LLMClient):

    engine = LLMEngine()
    engine.add_node("extracter", node=ExtracterNode())
    engine.add_node("prompts",
                    inputs=["/extracter"],
                    node=PromptTemplateNode(template="What is the capital of {{country}}?", template_format="jinja"))
    engine.add_node("completion", inputs=["/prompts"], node=LLMGenerateNode(llm_client=llm_client))
    engine.add_task_handler(inputs=["/completion"], handler=SimpleTaskHandler())

    return engine


def _run_pipeline(config: Config, llm_client: LLMClient, countries: list[str], capital_responses: list[str]) -> dict:
    """
    Loosely patterned after `examples/llm/completion`
    """
    source_df = cudf.DataFrame({"country": countries})
    expected_df = cudf.DataFrame({"country": countries, "response": capital_responses})

    completion_task = {"task_type": "completion", "task_dict": {"input_keys": ["country"]}}

    pipe = LinearPipeline(config)

    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))

    deserialize_config = config
    pipe.add_stage(DeserializeStage(deserialize_config, task_type="llm_engine", task_payload=completion_task))

    pipe.add_stage(LLMEngineStage(config, engine=_build_engine(llm_client)))

    sink = pipe.add_stage(CompareDataFrameStage(config, compare_df=expected_df))

    pipe.run()

    return sink.get_results()


@pytest.mark.usefixtures("nemollm")
def test_completion_pipe_nemo(config: Config,
                              mock_nemollm: mock.MagicMock,
                              countries: list[str],
                              capital_responses: list[str]):

    # Set a dummy key to bypass the API key check
    with set_env(NGC_API_KEY="test"):

        llm_client = NeMoLLMService().get_client(model_name="test_model")

        mock_nemollm.post_process_generate_response.side_effect = [{"text": response} for response in capital_responses]
        results = _run_pipeline(config, llm_client, countries=countries, capital_responses=capital_responses)
        assert_results(results)


@pytest.mark.usefixtures("openai")
def test_completion_pipe_openai(config: Config,
                                mock_chat_completion: tuple[mock.MagicMock, mock.MagicMock],
                                countries: list[str],
                                capital_responses: list[str]):
    (mock_client, mock_async_client) = mock_chat_completion
    mock_async_client.chat.completions.create.side_effect = [
        mk_mock_openai_response([response]) for response in capital_responses
    ]

    with set_env(OPENAI_API_KEY="test"):
        llm_client = OpenAIChatService().get_client(model_name="test_model")

        results = _run_pipeline(config, llm_client, countries=countries, capital_responses=capital_responses)
        assert_results(results)
        mock_client.chat.completions.create.assert_not_called()
        mock_async_client.chat.completions.create.assert_called()


@pytest.mark.usefixtures("nemollm")
@pytest.mark.usefixtures("ngc_api_key")
def test_completion_pipe_integration_nemo(config: Config, countries: list[str], capital_responses: list[str]):
    llm_client = NeMoLLMService().get_client(model_name="gpt-43b-002")

    results = _run_pipeline(config, llm_client, countries=countries, capital_responses=capital_responses)
    assert results['diff_cols'] == 0
    assert results['total_rows'] == len(countries)
    assert results['matching_rows'] + results['diff_rows'] == len(countries)


@pytest.mark.usefixtures("openai")
@pytest.mark.usefixtures("openai_api_key")
def test_completion_pipe_integration_openai(config: Config, countries: list[str], capital_responses: list[str]):
    llm_client = OpenAIChatService().get_client(model_name="gpt-3.5-turbo")

    results = _run_pipeline(config, llm_client, countries=countries, capital_responses=capital_responses)
    assert results['diff_cols'] == 0
    assert results['total_rows'] == len(countries)
    assert results['matching_rows'] + results['diff_rows'] == len(countries)
