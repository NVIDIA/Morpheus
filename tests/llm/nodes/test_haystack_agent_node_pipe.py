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

from unittest import mock

import pytest

from _utils import assert_results
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.llm import LLMEngine
from morpheus.llm.nodes.extracter_node import ExtracterNode
from morpheus.llm.nodes.haystack_agent_node import HaystackAgentNode
from morpheus.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus.messages import ControlMessage
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.llm.llm_engine_stage import LLMEngineStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage


def _build_engine(mock_haystack_agent: mock.MagicMock) -> LLMEngine:
    engine = LLMEngine()
    engine.add_node("extracter", node=ExtracterNode())
    engine.add_node("chain", inputs=["/extracter"], node=HaystackAgentNode(agent=mock_haystack_agent))
    engine.add_task_handler(inputs=["/chain"], handler=SimpleTaskHandler())

    return engine


@pytest.mark.use_python
def test_pipeline(config: Config,
                  dataset_cudf: DatasetManager,
                  mock_haystack_agent: mock.MagicMock,
                  mock_haystack_answer_data: dict,
                  mock_haystack_agent_run_return: dict):
    input_df = dataset_cudf["filter_probs.csv"]
    expected_df = input_df.copy(deep=True)

    mock_haystack_agent.run.return_value = mock_haystack_agent_run_return
    expected_df['response'] = [{'0': [mock_haystack_answer_data['answer']]} for _ in range(len(expected_df))]
    expected_calls = [mock.call({"query": x}) for x in expected_df['v3'].values_host]

    task_payload = {"task_type": "llm_engine", "task_dict": {"input_keys": ['v3']}}

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[input_df]))
    pipe.add_stage(
        DeserializeStage(config, message_type=ControlMessage, task_type="llm_engine", task_payload=task_payload))
    pipe.add_stage(LLMEngineStage(config, engine=_build_engine(mock_haystack_agent)))
    sink = pipe.add_stage(CompareDataFrameStage(config, compare_df=expected_df))

    pipe.run()

    assert_results(sink.get_results())
    mock_haystack_agent.run.assert_has_calls(expected_calls)
