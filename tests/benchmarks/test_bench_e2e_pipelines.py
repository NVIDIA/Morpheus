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

import json
import logging
import os

import pytest

from _utils import TEST_DIRS
from morpheus.config import Config
from morpheus.config import ConfigFIL
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.add_classifications_stage import AddClassificationsStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_fil_stage import PreprocessFILStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage
from morpheus.utils.file_utils import load_labels_file
from morpheus.utils.logger import set_log_level

E2E_CONFIG_FILE = os.path.join(TEST_DIRS.morpheus_root, "tests/benchmarks/e2e_test_configs.json")
with open(E2E_CONFIG_FILE, 'r', encoding='UTF-8') as f:
    E2E_TEST_CONFIGS = json.load(f)


def nlp_pipeline(config: Config, input_file, repeat, vocab_hash_file, output_file, model_name):

    set_log_level(log_level=logging.DEBUG)

    pipeline = LinearPipeline(config)
    pipeline.set_source(FileSourceStage(config, filename=input_file, repeat=repeat))
    pipeline.add_stage(DeserializeStage(config))
    pipeline.add_stage(
        PreprocessNLPStage(config,
                           vocab_hash_file=vocab_hash_file,
                           truncation=True,
                           do_lower_case=True,
                           add_special_tokens=False))
    pipeline.add_stage(
        TritonInferenceStage(config,
                             model_name=model_name,
                             server_url=E2E_TEST_CONFIGS["triton_server_url"],
                             force_convert_inputs=True))
    pipeline.add_stage(AddClassificationsStage(config, threshold=0.5, prefix=""))
    pipeline.add_stage(MonitorStage(config, log_level=logging.INFO))
    pipeline.add_stage(SerializeStage(config))
    pipeline.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))

    pipeline.build()
    pipeline.run()


def fil_pipeline(config: Config, input_file, repeat, output_file, model_name):

    set_log_level(log_level=logging.DEBUG)

    pipeline = LinearPipeline(config)
    pipeline.set_source(FileSourceStage(config, filename=input_file, repeat=repeat))
    pipeline.add_stage(DeserializeStage(config))
    pipeline.add_stage(PreprocessFILStage(config))
    pipeline.add_stage(
        TritonInferenceStage(config,
                             model_name=model_name,
                             server_url=E2E_TEST_CONFIGS["triton_server_url"],
                             force_convert_inputs=True))
    pipeline.add_stage(AddClassificationsStage(config, threshold=0.5, prefix=""))
    pipeline.add_stage(MonitorStage(config, log_level=logging.INFO))
    pipeline.add_stage(SerializeStage(config))
    pipeline.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))

    pipeline.build()
    pipeline.run()


@pytest.mark.benchmark
def test_sid_nlp_e2e(benchmark, tmp_path):

    config = Config()
    config.mode = PipelineModes.NLP
    config.num_threads = E2E_TEST_CONFIGS["test_sid_nlp_e2e"]["num_threads"]
    config.pipeline_batch_size = E2E_TEST_CONFIGS["test_sid_nlp_e2e"]["pipeline_batch_size"]
    config.model_max_batch_size = E2E_TEST_CONFIGS["test_sid_nlp_e2e"]["model_max_batch_size"]
    config.feature_length = E2E_TEST_CONFIGS["test_sid_nlp_e2e"]["feature_length"]
    config.edge_buffer_size = E2E_TEST_CONFIGS["test_sid_nlp_e2e"]["edge_buffer_size"]
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
    CppConfig.set_should_use_cpp(True)

    input_filepath = E2E_TEST_CONFIGS["test_sid_nlp_e2e"]["file_path"]
    repeat = E2E_TEST_CONFIGS["test_sid_nlp_e2e"]["repeat"]
    vocab_filepath = os.path.join(TEST_DIRS.data_dir, 'bert-base-uncased-hash.txt')
    output_filepath = os.path.join(tmp_path, "sid_nlp_e2e_output.csv")
    model_name = "sid-minibert-onnx"

    benchmark(nlp_pipeline, config, input_filepath, repeat, vocab_filepath, output_filepath, model_name)


@pytest.mark.benchmark
def test_abp_fil_e2e(benchmark, tmp_path):

    config = Config()
    config.mode = PipelineModes.FIL
    config.num_threads = E2E_TEST_CONFIGS["test_abp_fil_e2e"]["num_threads"]
    config.pipeline_batch_size = E2E_TEST_CONFIGS["test_abp_fil_e2e"]["pipeline_batch_size"]
    config.model_max_batch_size = E2E_TEST_CONFIGS["test_abp_fil_e2e"]["model_max_batch_size"]
    config.feature_length = E2E_TEST_CONFIGS["test_abp_fil_e2e"]["feature_length"]
    config.edge_buffer_size = E2E_TEST_CONFIGS["test_abp_fil_e2e"]["edge_buffer_size"]
    config.class_labels = ["mining"]
    config.fil = ConfigFIL()
    fil_cols_filepath = os.path.join(TEST_DIRS.data_dir, 'columns_fil.txt')
    config.fil.feature_columns = load_labels_file(fil_cols_filepath)
    CppConfig.set_should_use_cpp(True)

    input_filepath = E2E_TEST_CONFIGS["test_abp_fil_e2e"]["file_path"]
    repeat = E2E_TEST_CONFIGS["test_abp_fil_e2e"]["repeat"]
    output_filepath = os.path.join(tmp_path, "abp_fil_e2e_output.csv")
    model_name = "abp-nvsmi-xgb"

    benchmark(fil_pipeline, config, input_filepath, repeat, output_filepath, model_name)


@pytest.mark.benchmark
def test_phishing_nlp_e2e(benchmark, tmp_path):

    config = Config()
    config.mode = PipelineModes.NLP
    config.num_threads = E2E_TEST_CONFIGS["test_phishing_nlp_e2e"]["num_threads"]
    config.pipeline_batch_size = E2E_TEST_CONFIGS["test_phishing_nlp_e2e"]["pipeline_batch_size"]
    config.model_max_batch_size = E2E_TEST_CONFIGS["test_phishing_nlp_e2e"]["model_max_batch_size"]
    config.feature_length = E2E_TEST_CONFIGS["test_phishing_nlp_e2e"]["feature_length"]
    config.edge_buffer_size = E2E_TEST_CONFIGS["test_phishing_nlp_e2e"]["edge_buffer_size"]
    config.class_labels = load_labels_file(os.path.join(TEST_DIRS.data_dir, "labels_phishing.txt"))
    CppConfig.set_should_use_cpp(True)

    input_filepath = E2E_TEST_CONFIGS["test_phishing_nlp_e2e"]["file_path"]
    repeat = E2E_TEST_CONFIGS["test_phishing_nlp_e2e"]["repeat"]
    vocab_filepath = os.path.join(TEST_DIRS.data_dir, 'bert-base-uncased-hash.txt')
    output_filepath = os.path.join(tmp_path, "phishing_nlp_e2e_output.csv")
    model_name = "phishing-bert-onnx"

    benchmark(nlp_pipeline, config, input_filepath, repeat, vocab_filepath, output_filepath, model_name)
