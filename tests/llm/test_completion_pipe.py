# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import copy
from unittest import mock

import pytest

import cudf

from _utils import assert_results
from morpheus.config import Config
from morpheus.llm import LLMEngine
from morpheus.llm.nodes.extracter_node import ExtracterNode
from morpheus.llm.nodes.llm_generate_node import LLMGenerateNode
from morpheus.llm.nodes.prompt_template_node import PromptTemplateNode
from morpheus.llm.services.llm_service import LLMService
from morpheus.llm.services.nemo_llm_service import NeMoLLMService
from morpheus.llm.services.openai_chat_service import OpenAIChatService
from morpheus.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus.messages import ControlMessage
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.llm.llm_engine_stage import LLMEngineStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage


def _build_engine(llm_service_cls: LLMService, model_name: str = "test_model"):
    llm_service = llm_service_cls()
    llm_client = llm_service.get_client(model_name=model_name)

    engine = LLMEngine()
    engine.add_node("extracter", node=ExtracterNode())
    engine.add_node("prompts",
                    inputs=["/extracter"],
                    node=PromptTemplateNode(template="What is the capital of {{country}}?", template_format="jinja"))
    engine.add_node("completion", inputs=["/prompts"], node=LLMGenerateNode(llm_client=llm_client))
    engine.add_task_handler(inputs=["/completion"], handler=SimpleTaskHandler())

    return engine


def _run_pipeline(config: Config,
                  llm_service_cls: LLMService,
                  countries: list[str],
                  capital_responses: list[str],
                  model_name: str = "test_model") -> dict:
    """
    Loosely patterned after `examples/llm/completion`
    """
    source_df = cudf.DataFrame({"country": countries})
    expected_df = cudf.DataFrame({"country": countries, "response": capital_responses})

    completion_task = {"task_type": "completion", "task_dict": {"input_keys": ["country"]}}

    pipe = LinearPipeline(config)

    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))

    pipe.add_stage(
        DeserializeStage(config, message_type=ControlMessage, task_type="llm_engine", task_payload=completion_task))

    pipe.add_stage(LLMEngineStage(config, engine=_build_engine(llm_service_cls, model_name=model_name)))
    sink = pipe.add_stage(CompareDataFrameStage(config, compare_df=expected_df))

    pipe.run()

    return sink.get_results()


@pytest.mark.usefixtures("nemollm")
@pytest.mark.use_python
@mock.patch("asyncio.wrap_future")
@mock.patch("asyncio.gather", new_callable=mock.AsyncMock)
def test_completion_pipe_nemo(
        mock_asyncio_gather: mock.AsyncMock,
        mock_asyncio_wrap_future: mock.MagicMock,  # pylint: disable=unused-argument
        config: Config,
        mock_nemollm: mock.MagicMock,
        countries: list[str],
        capital_responses: list[str]):
    mock_asyncio_gather.return_value = [mock.MagicMock() for _ in range(len(countries))]
    mock_nemollm.post_process_generate_response.side_effect = [{"text": response} for response in capital_responses]
    results = _run_pipeline(config, NeMoLLMService, countries=countries, capital_responses=capital_responses)
    assert_results(results)


@pytest.mark.usefixtures("openai")
@pytest.mark.use_python
def test_completion_pipe_openai(config: Config,
                                chat_completion,
                                mock_async_openai: mock.MagicMock,
                                countries: list[str],
                                capital_responses: list[str]):

    chat_completions = []
    for response in capital_responses:
        chat_completion_cp = copy.deepcopy(chat_completion)
        chat_completion_cp.choices[0].message.content = response
        chat_completions.append(chat_completion_cp)

    mock_async_openai.chat.completions.create.side_effect = chat_completions

    results = _run_pipeline(config, OpenAIChatService, countries=countries, capital_responses=capital_responses)
    assert_results(results)


@pytest.mark.usefixtures("nemollm")
@pytest.mark.usefixtures("ngc_api_key")
@pytest.mark.use_python
def test_completion_pipe_integration_nemo(config: Config, countries: list[str], capital_responses: list[str]):
    results = _run_pipeline(config,
                            NeMoLLMService,
                            countries=countries,
                            capital_responses=capital_responses,
                            model_name="gpt-43b-002")
    assert results['diff_cols'] == 0
    assert results['total_rows'] == len(countries)
    assert results['matching_rows'] + results['diff_rows'] == len(countries)


@pytest.mark.usefixtures("openai")
@pytest.mark.usefixtures("openai_api_key")
@pytest.mark.use_python
def test_completion_pipe_integration_openai(config: Config, countries: list[str], capital_responses: list[str]):
    results = _run_pipeline(config,
                            OpenAIChatService,
                            countries=countries,
                            capital_responses=capital_responses,
                            model_name="gpt-3.5-turbo")
    assert results['diff_cols'] == 0
    assert results['total_rows'] == len(countries)
    assert results['matching_rows'] + results['diff_rows'] == len(countries)
