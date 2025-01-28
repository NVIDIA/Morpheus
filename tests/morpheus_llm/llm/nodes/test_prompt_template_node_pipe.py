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

import pytest

import cudf

from _utils import assert_results
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus_llm.llm import LLMEngine
from morpheus_llm.llm.nodes.extracter_node import ExtracterNode
from morpheus_llm.llm.nodes.prompt_template_node import PromptTemplateNode
from morpheus_llm.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus_llm.stages.llm.llm_engine_stage import LLMEngineStage

MULTI_LINE_JINJA_TEMPLATE = """Testing a loop:
{% for lv in ctx.list_values -%}Title: {{ lv.title }}, Summary: {{ lv.summary }}
{% endfor %}
{{ ctx.query }}"""


def _build_engine(template: str, template_format: str) -> LLMEngine:
    engine = LLMEngine()
    engine.add_node("extracter", node=ExtracterNode())
    engine.add_node("prompts",
                    inputs=["/extracter"],
                    node=PromptTemplateNode(template=template, template_format=template_format))

    engine.add_task_handler(inputs=["/prompts"], handler=SimpleTaskHandler())

    return engine


@pytest.mark.parametrize("template,template_format,values,expected_output",
                         [("Hello {name}!",
                           "f-string", {
                               'name': ['World', 'Universe', 'Galaxy', 'Moon']
                           }, ["Hello World!", "Hello Universe!", "Hello Galaxy!", "Hello Moon!"]),
                          ("Hello {{ name }}!",
                           "jinja", {
                               'name': ['World', 'Universe', 'Galaxy', 'Moon']
                           }, ["Hello World!", "Hello Universe!", "Hello Galaxy!", "Hello Moon!"])],
                         ids=["f-string-hello-world", "jinja-hello-world"])
def test_prompt_template_node_pipe(config: Config,
                                   template: str,
                                   template_format: str,
                                   values: dict,
                                   expected_output: list[str]):
    config.mode = PipelineModes.OTHER
    input_df = cudf.DataFrame(values)
    expected_df = input_df.copy(deep=True)
    expected_df["response"] = expected_output

    input_names = sorted(values.keys())
    task_payload = {"task_type": "llm_engine", "task_dict": {"input_keys": input_names}}

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[input_df]))
    pipe.add_stage(DeserializeStage(config, task_type="llm_engine", task_payload=task_payload))
    pipe.add_stage(LLMEngineStage(config, engine=_build_engine(template=template, template_format=template_format)))
    sink = pipe.add_stage(CompareDataFrameStage(config, compare_df=expected_df))

    pipe.run()

    assert_results(sink.get_results())
