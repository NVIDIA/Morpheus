#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from _utils import TEST_DIRS
from _utils import calc_error_val
from _utils import compare_class_to_scores
from _utils import mk_async_infer
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.messages import MultiMessage
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.add_classifications_stage import AddClassificationsStage
from morpheus.stages.postprocess.add_scores_stage import AddScoresStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.postprocess.validation_stage import ValidationStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage

# End-to-end test intended to imitate the Sid validation test
FEATURE_LENGTH = 256
MODEL_MAX_BATCH_SIZE = 32


def _run_minibert_pipeline(
    *,
    config: Config,
    tmp_path: str,
    message_type: type,
    model_name: str,
    truncated: bool,
    morpheus_log_level: int,
    data_col_name: str = "data",
):
    """
    Runs just the Minibert Pipeline
    """

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

    val_file_name = os.path.join(TEST_DIRS.validation_data_dir, 'sid-validation-data.csv')
    vocab_file_name = os.path.join(TEST_DIRS.data_dir, 'bert-base-uncased-hash.txt')
    out_file = os.path.join(tmp_path, 'results.csv')
    results_file_name = os.path.join(tmp_path, 'results.json')

    # Create an augumented val file with the data column changed
    if (data_col_name != "data"):
        # Read the val file in
        val_df = pd.read_csv(val_file_name, index_col=0)

        # Change the column name
        val_df.rename(columns={"data": data_col_name}, inplace=True)

        # Now write to a temp file
        new_val_file = os.path.join(tmp_path, "augumented-sid-validation-data.csv")
        val_df.to_csv(new_val_file)

        # Use the new validation filename
        val_file_name = new_val_file

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=val_file_name, iterative=False))
    pipe.add_stage(DeserializeStage(config, message_type=message_type))
    pipe.add_stage(
        PreprocessNLPStage(config,
                           vocab_hash_file=vocab_file_name,
                           truncation=truncated,
                           do_lower_case=True,
                           add_special_tokens=False,
                           column=data_col_name))
    pipe.add_stage(
        TritonInferenceStage(config, model_name=model_name, server_url='localhost:8001', force_convert_inputs=True))
    pipe.add_stage(
        MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf", log_level=morpheus_log_level))
    pipe.add_stage(AddClassificationsStage(config, threshold=0.5, prefix="si_"))
    pipe.add_stage(AddScoresStage(config, prefix="score_"))
    pipe.add_stage(
        ValidationStage(config, val_file_name=val_file_name, results_file_name=results_file_name, rel_tol=0.05))
    pipe.add_stage(SerializeStage(config))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))

    pipe.run()
    compare_class_to_scores(out_file, config.class_labels, 'si_', 'score_', threshold=0.5)
    return calc_error_val(results_file_name)


def _run_minibert(*,
                  config: Config,
                  tmp_path: str,
                  message_type: type,
                  model_name: str,
                  truncated: bool,
                  morpheus_log_level: int,
                  data_col_name: str = "data"):
    """
    Runs the minibert pipeline and mocks the Triton Python interface
    """

    # Setup the python mocking for Triton if necessary. Wont be used if we are C++
    with mock.patch('tritonclient.grpc.InferenceServerClient') as mock_triton_client:
        mock_metadata = {
            "inputs": [{
                "name": "input_ids", "datatype": "INT32", "shape": [-1, FEATURE_LENGTH]
            }, {
                "name": "attention_mask", "datatype": "INT32", "shape": [-1, FEATURE_LENGTH]
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

        data = np.loadtxt(os.path.join(TEST_DIRS.tests_data_dir, 'triton_sid_inf_results.csv'), delimiter=',')
        inf_results = np.split(data, range(MODEL_MAX_BATCH_SIZE, len(data), MODEL_MAX_BATCH_SIZE))

        async_infer = mk_async_infer(inf_results)
        mock_triton_client.async_infer.side_effect = async_infer

        return _run_minibert_pipeline(config=config,
                                      tmp_path=tmp_path,
                                      message_type=message_type,
                                      model_name=model_name,
                                      truncated=truncated,
                                      data_col_name=data_col_name,
                                      morpheus_log_level=morpheus_log_level)


@pytest.mark.slow
@pytest.mark.use_cpp
@pytest.mark.usefixtures("launch_mock_triton")
@pytest.mark.parametrize("message_type", [MultiMessage, ControlMessage])
def test_minibert_no_trunc(config: Config, tmp_path: str, message_type: type, morpheus_log_level: int):

    config.num_threads = os.cpu_count() or 1

    results = _run_minibert(config=config,
                            tmp_path=tmp_path,
                            message_type=message_type,
                            model_name="sid-minibert-onnx-no-trunc",
                            truncated=False,
                            morpheus_log_level=morpheus_log_level)

    # Not sure why these are different
    if (CppConfig.get_should_use_cpp()):
        assert results.diff_rows == 18
    else:
        assert results.diff_rows == 1333


@pytest.mark.slow
@pytest.mark.usefixtures("launch_mock_triton")
@pytest.mark.parametrize("data_col_name", ["data", "definitely_not_data"])
@pytest.mark.parametrize("message_type", [MultiMessage, ControlMessage])
def test_minibert_truncated(config: Config,
                            tmp_path: str,
                            message_type: type,
                            morpheus_log_level: int,
                            data_col_name: str):

    results = _run_minibert(config=config,
                            tmp_path=tmp_path,
                            message_type=message_type,
                            model_name='sid-minibert-onnx',
                            truncated=True,
                            data_col_name=data_col_name,
                            morpheus_log_level=morpheus_log_level)

    # Not sure why these are different
    if (CppConfig.get_should_use_cpp()):
        assert results.diff_rows == 1204
    else:
        assert results.diff_rows == 1333
