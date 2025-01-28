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

import collections.abc
import os
import typing

import pytest

import cudf

from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus_llm.llm import LLMEngine
from morpheus_llm.llm.nodes.extracter_node import ExtracterNode
from morpheus_llm.llm.nodes.llm_generate_node import LLMGenerateNode
from morpheus_llm.llm.nodes.prompt_template_node import PromptTemplateNode
from morpheus_llm.llm.services.llm_service import LLMService
from morpheus_llm.llm.services.nemo_llm_service import NeMoLLMService
from morpheus_llm.llm.services.openai_chat_service import OpenAIChatService
from morpheus_llm.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus_llm.stages.llm.llm_engine_stage import LLMEngineStage


def _build_engine(llm_service_cls: type[LLMService], model_name: str = "test_model"):
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
                  llm_service_cls: type[LLMService],
                  source_df: cudf.DataFrame,
                  model_name: str = "test_model") -> dict:
    """
    Loosely patterned after `examples/llm/completion`
    """
    completion_task = {"task_type": "completion", "task_dict": {"input_keys": ["country"]}}

    pipe = LinearPipeline(config)

    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))

    pipe.add_stage(DeserializeStage(config, task_type="llm_engine", task_payload=completion_task))

    pipe.add_stage(LLMEngineStage(config, engine=_build_engine(llm_service_cls, model_name=model_name)))
    pipe.add_stage(InMemorySinkStage(config))

    pipe.run()


@pytest.mark.use_cudf
@pytest.mark.cpu_mode
@pytest.mark.benchmark
@pytest.mark.usefixtures("mock_nemollm", "mock_chat_completion")
@pytest.mark.parametrize("llm_service_cls", [NeMoLLMService, OpenAIChatService])
def test_completion_pipe(benchmark: collections.abc.Callable[[collections.abc.Callable], typing.Any],
                         config: Config,
                         dataset: DatasetManager,
                         llm_service_cls: type[LLMService]):
    if llm_service_cls == OpenAIChatService:
        os.environ.update({"OPENAI_API_KEY": "test_api_key"})
    benchmark(_run_pipeline, config, llm_service_cls, source_df=dataset["countries.csv"])
