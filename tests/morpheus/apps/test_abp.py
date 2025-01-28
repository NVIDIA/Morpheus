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

import pytest

from _utils import TEST_DIRS
from _utils import calc_error_val
from _utils import compare_class_to_scores
from morpheus.config import Config
from morpheus.config import ConfigFIL
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
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
from morpheus.stages.preprocess.preprocess_fil_stage import PreprocessFILStage
from morpheus.utils.file_utils import load_labels_file

# End-to-end test intended to imitate the ABP validation test
FEATURE_LENGTH = 18
MODEL_MAX_BATCH_SIZE = 1024


@pytest.mark.slow
@pytest.mark.gpu_mode
@pytest.mark.usefixtures("launch_mock_triton")
def test_abp_cpp(config: Config, tmp_path: str, morpheus_log_level: int):
    config.mode = PipelineModes.FIL
    config.class_labels = ["mining"]
    config.model_max_batch_size = MODEL_MAX_BATCH_SIZE
    config.pipeline_batch_size = 1024
    config.feature_length = FEATURE_LENGTH
    config.edge_buffer_size = 128
    config.num_threads = 1

    config.fil = ConfigFIL()
    config.fil.feature_columns = load_labels_file(os.path.join(TEST_DIRS.data_dir, 'columns_fil.txt'))

    val_file_name = os.path.join(TEST_DIRS.validation_data_dir, 'abp-validation-data.jsonlines')
    out_file = os.path.join(tmp_path, 'results.csv')
    results_file_name = os.path.join(tmp_path, 'results.json')

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=val_file_name, iterative=False))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(PreprocessFILStage(config))

    # We are feeding TritonInferenceStage the port to the grpc server because that is what the validation tests do
    # but the code under-the-hood replaces this with the port number of the http server
    pipe.add_stage(
        TritonInferenceStage(config, model_name='abp-nvsmi-xgb', server_url='localhost:8001',
                             force_convert_inputs=True))
    pipe.add_stage(
        MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf", log_level=morpheus_log_level))
    pipe.add_stage(AddClassificationsStage(config))
    pipe.add_stage(AddScoresStage(config, prefix="score_"))
    pipe.add_stage(
        ValidationStage(config,
                        val_file_name=val_file_name,
                        results_file_name=results_file_name,
                        rel_tol=0.05,
                        overwrite=True))
    pipe.add_stage(SerializeStage(config))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=True))

    pipe.run()

    compare_class_to_scores(out_file, config.class_labels, '', 'score_', threshold=0.5)
    results = calc_error_val(results_file_name)
    assert results.diff_rows == 0


@pytest.mark.slow
@pytest.mark.gpu_mode
@pytest.mark.usefixtures("launch_mock_triton")
def test_abp_multi_segment_cpp(config, tmp_path):

    config.mode = PipelineModes.FIL
    config.class_labels = ["mining"]
    config.model_max_batch_size = MODEL_MAX_BATCH_SIZE
    config.pipeline_batch_size = 1024
    config.feature_length = FEATURE_LENGTH
    config.edge_buffer_size = 128
    config.num_threads = 1

    config.fil = ConfigFIL()
    config.fil.feature_columns = load_labels_file(os.path.join(TEST_DIRS.data_dir, 'columns_fil.txt'))

    val_file_name = os.path.join(TEST_DIRS.validation_data_dir, 'abp-validation-data.jsonlines')
    out_file = os.path.join(tmp_path, 'results.csv')
    results_file_name = os.path.join(tmp_path, 'results.json')

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=val_file_name, iterative=False))
    pipe.add_stage(DeserializeStage(config))

    pipe.add_segment_boundary(ControlMessage)  # Boundary 1

    pipe.add_stage(PreprocessFILStage(config))

    pipe.add_segment_boundary(ControlMessage)  # Boundary 2

    # We are feeding TritonInferenceStage the port to the grpc server because that is what the validation tests do
    # but the code under-the-hood replaces this with the port number of the http server
    pipe.add_stage(
        TritonInferenceStage(config, model_name='abp-nvsmi-xgb', server_url='localhost:8001',
                             force_convert_inputs=True))

    pipe.add_segment_boundary(ControlMessage)  # Boundary 3

    pipe.add_stage(MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf"))
    pipe.add_stage(AddClassificationsStage(config))

    pipe.add_segment_boundary(ControlMessage)  # Boundary 4

    pipe.add_stage(
        ValidationStage(config, val_file_name=val_file_name, results_file_name=results_file_name, rel_tol=0.05))

    pipe.add_segment_boundary(ControlMessage)  # Boundary 5

    pipe.add_stage(SerializeStage(config))

    pipe.add_segment_boundary(MessageMeta)  # Boundary 6

    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))

    pipe.run()

    results = calc_error_val(results_file_name)
    assert results.diff_rows == 0
