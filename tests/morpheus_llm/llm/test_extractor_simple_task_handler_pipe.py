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

from _utils import assert_results
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus_llm.llm import LLMEngine
from morpheus_llm.llm.nodes.extracter_node import ExtracterNode
from morpheus_llm.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus_llm.stages.llm.llm_engine_stage import LLMEngineStage


def _build_engine() -> LLMEngine:
    engine = LLMEngine()
    engine.add_node("extracter", node=ExtracterNode())
    engine.add_task_handler(inputs=["/extracter"], handler=SimpleTaskHandler())

    return engine


def test_extractor_simple_task_handler_pipeline(config: Config, dataset_cudf: DatasetManager):
    input_df = dataset_cudf["filter_probs.csv"]
    expected_df = input_df.copy(deep=True)
    expected_df['response'] = input_df['v3']

    task_payload = {"task_type": "llm_engine", "task_dict": {"input_keys": ['v3']}}

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[input_df]))
    pipe.add_stage(DeserializeStage(config, task_type="llm_engine", task_payload=task_payload))
    pipe.add_stage(LLMEngineStage(config, engine=_build_engine()))
    sink = pipe.add_stage(CompareDataFrameStage(config, compare_df=expected_df))

    pipe.run()

    assert_results(sink.get_results())
