#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import types
import typing

import pytest

from _utils.dataset_manager import DatasetManager
from _utils.kafka import KafkaTopics
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.write_to_kafka_stage import WriteToKafkaStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage

if (typing.TYPE_CHECKING):
    from kafka import KafkaConsumer


@pytest.mark.kafka
@pytest.mark.gpu_and_cpu_mode
def test_write_to_kafka_stage_pipe(config,
                                   df_pkg: types.ModuleType,
                                   dataset: DatasetManager,
                                   kafka_bootstrap_servers: str,
                                   kafka_consumer: "KafkaConsumer",
                                   kafka_topics: KafkaTopics) -> None:
    """
    Even though WriteToKafkaStage only has a Python impl, testing with both C++ and Python execution
    to ensure it works just as well with the C++ impls of the message classes.
    """

    filter_probs_df = dataset['filter_probs.csv']
    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(SerializeStage(config))
    pipe.add_stage(
        WriteToKafkaStage(config,
                          bootstrap_servers=kafka_bootstrap_servers,
                          output_topic=kafka_topics.output_topic,
                          client_id='morpheus_test_write_to_kafka_stage_pipe'))
    pipe.run()

    kafka_messages = list(kafka_consumer)
    assert len(kafka_messages) == len(filter_probs_df)

    output_df = df_pkg.read_json("\n".join(rec.value.decode("utf-8") for rec in kafka_messages), lines=True)

    assert len(output_df) == len(filter_probs_df)

    dataset.assert_compare_df(filter_probs_df, output_df)
