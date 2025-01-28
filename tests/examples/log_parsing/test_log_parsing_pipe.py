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
import types
import typing
from unittest import mock

import numpy as np
import pytest

from _utils import TEST_DIRS
from _utils import assert_results
from _utils import mk_async_infer
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage

FEATURE_LENGTH = 256
MODEL_MAX_BATCH_SIZE = 32


def _run_pipeline(config: Config,
                  dataset_cudf: DatasetManager,
                  import_mod: typing.List[types.ModuleType],
                  bert_cased_hash: str,
                  bert_cased_vocab: str):
    """
    Runs just the Log Parsing Pipeline
    """
    inference_mod, postprocessing_mod = import_mod

    config.num_threads = 1
    config.pipeline_batch_size = 1024
    config.model_max_batch_size = MODEL_MAX_BATCH_SIZE
    config.feature_length = FEATURE_LENGTH

    log_test_data_dir = os.path.join(TEST_DIRS.tests_data_dir, 'examples/log_parsing')

    # Not actually the real model config, just the subset that LogParsingPostProcessingStage uses
    model_config_file = os.path.join(log_test_data_dir, 'log-parsing-config.json')

    input_file = os.path.join(TEST_DIRS.validation_data_dir, 'log-parsing-validation-data-input.csv')
    input_df = dataset_cudf[input_file]

    # We only have expected results for the first 5 rows
    input_df = input_df[0:5]

    expected_df = dataset_cudf.pandas[os.path.join(log_test_data_dir, 'expected_out.csv')]

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [input_df]))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(
        PreprocessNLPStage(config,
                           vocab_hash_file=bert_cased_hash,
                           truncation=False,
                           do_lower_case=False,
                           stride=64,
                           add_special_tokens=False,
                           column="raw"))
    pipe.add_stage(
        inference_mod.LogParsingInferenceStage(config,
                                               model_name='log-parsing-onnx',
                                               server_url='localhost:8001',
                                               force_convert_inputs=True))
    pipe.add_stage(
        postprocessing_mod.LogParsingPostProcessingStage(config,
                                                         vocab_path=bert_cased_vocab,
                                                         model_config_path=model_config_file))

    comp_stage = pipe.add_stage(CompareDataFrameStage(config, expected_df))

    pipe.run()

    assert_results(comp_stage.get_results())


def _run_mocked_pipeline(config: Config,
                         dataset_cudf: DatasetManager,
                         import_mod: typing.List[types.ModuleType],
                         bert_cased_hash: str,
                         bert_cased_vocab: str):
    """
    Runs the minibert pipeline and mocks the Triton Python interface
    """

    # Setup the python mocking for Triton if necessary. Wont be used if we are C++
    with mock.patch('tritonclient.grpc.InferenceServerClient') as mock_triton_client:
        mock_metadata = {
            "inputs": [{
                "name": "input_ids", "datatype": "INT64", "shape": [-1, FEATURE_LENGTH]
            }, {
                "name": "attention_mask", "datatype": "INT64", "shape": [-1, FEATURE_LENGTH]
            }],
            "outputs": [{
                "name": "output", "datatype": "FP32", "shape": [-1, 10]
            }]
        }
        mock_model_config = {"config": {"max_batch_size": MODEL_MAX_BATCH_SIZE}}

        mock_triton_client.return_value = mock_triton_client
        mock_triton_client.is_server_live.return_value = True
        mock_triton_client.is_server_ready.return_value = True
        mock_triton_client.is_model_ready.return_value = True
        mock_triton_client.get_model_metadata.return_value = mock_metadata
        mock_triton_client.get_model_config.return_value = mock_model_config

        data = np.load(os.path.join(TEST_DIRS.tests_data_dir, 'examples/log_parsing/triton_results.npy'))
        inf_results = np.split(data, range(MODEL_MAX_BATCH_SIZE, len(data), MODEL_MAX_BATCH_SIZE))

        async_infer = mk_async_infer(inf_results)
        mock_triton_client.async_infer.side_effect = async_infer

        return _run_pipeline(config,
                             dataset_cudf,
                             import_mod,
                             bert_cased_hash=bert_cased_hash,
                             bert_cased_vocab=bert_cased_vocab)


@pytest.mark.slow
@pytest.mark.import_mod([
    os.path.join(TEST_DIRS.examples_dir, 'log_parsing', 'inference.py'),
    os.path.join(TEST_DIRS.examples_dir, 'log_parsing', 'postprocessing.py')
])
def test_pipe(config: Config,
              dataset_cudf: DatasetManager,
              import_mod: typing.List[types.ModuleType],
              bert_cased_hash: str,
              bert_cased_vocab: str):
    _run_mocked_pipeline(config,
                         dataset_cudf,
                         import_mod,
                         bert_cased_hash=bert_cased_hash,
                         bert_cased_vocab=bert_cased_vocab)
