#!/usr/bin/env python
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
from datetime import datetime

import pytest

from _utils import TEST_DIRS
from _utils import assert_results
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.pipeline.stage_decorator import stage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
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


@pytest.mark.gpu_and_cpu_mode
def test_pipeline(config: Config, dataset: DatasetManager):
    test_data = os.path.join(TEST_DIRS.validation_data_dir, 'root-cause-validation-data-input.jsonlines')
    input_df = dataset[test_data]
    expected_df = input_df.copy(deep=True)
    expected_df["response"] = expected_df['log']

    task_payload = {"task_type": "llm_engine", "task_dict": {"input_keys": ['log']}}
    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[input_df]))
    pipe.add_stage(DeserializeStage(config, task_type="llm_engine", task_payload=task_payload))
    pipe.add_stage(LLMEngineStage(config, engine=_build_engine()))
    sink = pipe.add_stage(CompareDataFrameStage(config, compare_df=expected_df))

    pipe.run()

    assert_results(sink.get_results())


@pytest.mark.gpu_and_cpu_mode
def test_error_1973(config: Config, dataset: DatasetManager):
    expected_timestamps: dict[str, datetime] = {}

    @stage(execution_modes=(config.execution_mode, ))
    def log_timestamp(msg: ControlMessage, *, timestamp_name: str) -> ControlMessage:
        ts = datetime.now()
        msg.set_timestamp(key=timestamp_name, timestamp=ts)
        expected_timestamps[timestamp_name] = ts
        return msg

    task_payload = {"task_type": "llm_engine", "task_dict": {"input_keys": ['v1']}}
    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[dataset["filter_probs.csv"]]))
    pipe.add_stage(DeserializeStage(config, task_type="llm_engine", task_payload=task_payload))
    pipe.add_stage(log_timestamp(config, timestamp_name="pre_llm"))
    pipe.add_stage(LLMEngineStage(config, engine=_build_engine()))
    pipe.add_stage(log_timestamp(config, timestamp_name="post_llm"))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()
    assert len(messages) == 1

    msg = messages[0]
    for (timestamp_name, expected_timestamp) in expected_timestamps.items():
        actual_timestamp = msg.get_timestamp(timestamp_name, fail_if_nonexist=True)
        assert actual_timestamp == expected_timestamp
