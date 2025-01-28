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

import os
import typing

import pandas as pd
import pytest

from _utils import TEST_DIRS
from _utils import assert_results
from _utils.kafka import KafkaTopics
from _utils.kafka import seek_to_beginning
from _utils.kafka import write_data_to_kafka
from _utils.kafka import write_file_to_kafka
from _utils.stages.dfp_length_checker import DFPLengthChecker
from morpheus.config import Config
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.trigger_stage import TriggerStage
from morpheus.stages.input.kafka_source_stage import AutoOffsetReset
from morpheus.stages.input.kafka_source_stage import KafkaSourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage

if (typing.TYPE_CHECKING):
    from kafka import KafkaConsumer


@pytest.mark.gpu_and_cpu_mode
@pytest.mark.kafka
def test_kafka_source_stage_pipe(config: Config, kafka_bootstrap_servers: str, kafka_topics: KafkaTopics) -> None:
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.jsonlines")

    # Fill our topic with the input data
    num_records = write_file_to_kafka(kafka_bootstrap_servers, kafka_topics.input_topic, input_file)

    pipe = LinearPipeline(config)
    pipe.set_source(
        KafkaSourceStage(config,
                         bootstrap_servers=kafka_bootstrap_servers,
                         input_topic=kafka_topics.input_topic,
                         auto_offset_reset=AutoOffsetReset.EARLIEST,
                         poll_interval="1seconds",
                         client_id='morpheus_kafka_source_stage_pipe',
                         stop_after=num_records))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(SerializeStage(config))
    comp_stage = pipe.add_stage(CompareDataFrameStage(config, input_file))
    pipe.run()

    assert_results(comp_stage.get_results())


@pytest.mark.gpu_and_cpu_mode
@pytest.mark.kafka
def test_multi_topic_kafka_source_stage_pipe(config: Config, kafka_bootstrap_servers: str) -> None:
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.jsonlines")

    topic_1 = "morpheus_input_topic_1"
    topic_2 = "morpheus_input_topic_2"

    input_topics = [topic_1, topic_2]

    # Fill our topic_1 and topic_2 with the input data
    topic_1_records = write_file_to_kafka(kafka_bootstrap_servers, topic_1, input_file)
    topic_2_records = write_file_to_kafka(kafka_bootstrap_servers, topic_2, input_file)

    num_records = topic_1_records + topic_2_records

    pipe = LinearPipeline(config)
    pipe.set_source(
        KafkaSourceStage(config,
                         bootstrap_servers=kafka_bootstrap_servers,
                         input_topic=input_topics,
                         auto_offset_reset=AutoOffsetReset.EARLIEST,
                         poll_interval="1seconds",
                         client_id='test_multi_topic_kafka_source_stage_pipe',
                         stop_after=num_records))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(SerializeStage(config))
    comp_stage = pipe.add_stage(CompareDataFrameStage(config, [input_file, input_file]))
    pipe.run()

    assert_results(comp_stage.get_results())


@pytest.mark.gpu_and_cpu_mode
@pytest.mark.kafka
@pytest.mark.parametrize('async_commits', [True, False])
@pytest.mark.parametrize('num_records', [10, 1000])
def test_kafka_source_batch_pipe(num_records: int,
                                 async_commits: bool,
                                 config: Config,
                                 kafka_bootstrap_servers: str,
                                 kafka_topics: KafkaTopics,
                                 kafka_consumer: "KafkaConsumer") -> None:
    group_id = 'morpheus'

    data = [{'v': i} for i in range(num_records)]
    num_written = write_data_to_kafka(kafka_bootstrap_servers, kafka_topics.input_topic, data)
    assert num_written == num_records

    expected_length = config.pipeline_batch_size
    num_exact = num_records // expected_length

    kafka_consumer.subscribe([kafka_topics.input_topic])
    seek_to_beginning(kafka_consumer)
    partitions = kafka_consumer.assignment()

    # This method does not advance the consumer, and this consumer has a different group_id than the source stage
    expected_offsets = kafka_consumer.end_offsets(partitions)

    pipe = LinearPipeline(config)
    pipe.set_source(
        KafkaSourceStage(config,
                         bootstrap_servers=kafka_bootstrap_servers,
                         input_topic=kafka_topics.input_topic,
                         auto_offset_reset=AutoOffsetReset.EARLIEST,
                         poll_interval="1seconds",
                         group_id=group_id,
                         client_id='morpheus_kafka_source_commit',
                         stop_after=num_records,
                         async_commits=async_commits))
    pipe.add_stage(DFPLengthChecker(config, expected_length=expected_length, num_exact=num_exact))
    pipe.add_stage(TriggerStage(config))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(SerializeStage(config))
    comp_stage = pipe.add_stage(
        CompareDataFrameStage(config, pd.DataFrame(data=data), include=[r'^v$'], reset_index=True))
    pipe.run()

    assert_results(comp_stage.get_results())

    from kafka import KafkaAdminClient
    admin_client = KafkaAdminClient(bootstrap_servers=kafka_bootstrap_servers, client_id='offset_checker')
    offsets = admin_client.list_consumer_group_offsets(group_id)

    # The broker may have created additional partitions, offsets should be a superset of expected_offsets
    for (topic_partition, expected_offset) in expected_offsets.items():
        # The value of the offsets dict being returned is a tuple of (offset, metadata), while the value of the
        # expected_offsets is just the offset.
        actual_offset = offsets[topic_partition][0]
        assert actual_offset == expected_offset
