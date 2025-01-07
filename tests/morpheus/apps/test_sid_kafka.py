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
from io import StringIO

import pandas
import pytest

from _utils import TEST_DIRS
from _utils.dataset_manager import DatasetManager
from _utils.kafka import KafkaTopics
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.io.utils import filter_null_data
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_kafka_stage import WriteToKafkaStage
from morpheus.stages.postprocess.add_classifications_stage import AddClassificationsStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage
from morpheus.utils.compare_df import compare_df

if (typing.TYPE_CHECKING):
    from kafka import KafkaConsumer

# End-to-end test intended to imitate the Sid validation test
FEATURE_LENGTH = 256
MODEL_MAX_BATCH_SIZE = 32


@pytest.mark.kafka
@pytest.mark.slow
@pytest.mark.gpu_mode
@pytest.mark.usefixtures("launch_mock_triton")
def test_minibert_cpp(dataset_pandas: DatasetManager,
                      config: Config,
                      kafka_bootstrap_servers: str,
                      kafka_topics: KafkaTopics,
                      kafka_consumer: "KafkaConsumer",
                      morpheus_log_level: int):
    config.mode = PipelineModes.NLP
    config.class_labels = [
        "address",
        "bank_acct",
        "credit_card",
        "email",
        "govt_id",
        "name",
        "password",
        "phone_num",
        "secret_keys",
        "user"
    ]
    config.model_max_batch_size = MODEL_MAX_BATCH_SIZE
    config.pipeline_batch_size = 1024
    config.feature_length = FEATURE_LENGTH
    config.edge_buffer_size = 128
    config.num_threads = 1

    val_file_name = os.path.join(TEST_DIRS.validation_data_dir, 'sid-validation-data.csv')
    vocab_file_name = os.path.join(TEST_DIRS.data_dir, 'bert-base-uncased-hash.txt')

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=val_file_name, iterative=False))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(
        PreprocessNLPStage(config,
                           vocab_hash_file=vocab_file_name,
                           truncation=True,
                           do_lower_case=True,
                           add_special_tokens=False))
    pipe.add_stage(
        TritonInferenceStage(config,
                             model_name='sid-minibert-onnx',
                             server_url='localhost:8001',
                             force_convert_inputs=True))
    pipe.add_stage(
        MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf", log_level=morpheus_log_level))
    pipe.add_stage(AddClassificationsStage(config, threshold=0.5, prefix="si_"))
    pipe.add_stage(SerializeStage(config))
    pipe.add_stage(
        WriteToKafkaStage(config, bootstrap_servers=kafka_bootstrap_servers, output_topic=kafka_topics.output_topic))

    pipe.run()

    val_df = dataset_pandas[val_file_name]

    output_buf = StringIO()
    for rec in kafka_consumer:
        output_buf.write(f"{rec.value.decode('utf-8')}\n")

    output_buf.seek(0)
    output_df = pandas.read_json(output_buf, lines=True)
    output_df = filter_null_data(output_df)

    assert len(output_df) == len(val_df)

    results = compare_df(val_df, output_df, exclude_columns=[r'^ID$', r'^_ts_'], rel_tol=0.05)

    assert results['diff_rows'] == 1204
