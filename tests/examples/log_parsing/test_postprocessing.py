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

import logging
import os
import re
import types
import typing

import cupy as cp
import numpy as np
import pytest

from _utils import TEST_DIRS
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.messages import TensorMemory


@pytest.fixture(scope='module', name="model_config_file")
def fixture_model_config_file():
    return os.path.join(TEST_DIRS.tests_data_dir, 'examples/log_parsing/log-parsing-config.json')


def build_post_proc_message(dataset_cudf: DatasetManager, log_test_data_dir: str):
    input_file = os.path.join(TEST_DIRS.validation_data_dir, 'log-parsing-validation-data-input.csv')

    # we have tensor data for the first five rows
    input_df = dataset_cudf[input_file][:5]
    meta = MessageMeta(input_df)

    count = 5
    tensors = {}
    for tensor_name in ['confidences', 'input_ids', 'labels']:
        tensor_file = os.path.join(log_test_data_dir, f'{tensor_name}.csv')
        host_data = np.loadtxt(tensor_file, delimiter=',')
        tensors[tensor_name] = cp.asarray(host_data)

    host__seq_data = np.loadtxt(os.path.join(log_test_data_dir, 'seq_ids.csv'), delimiter=',')
    seq_ids = cp.zeros((count, 3), dtype=cp.uint32)
    seq_ids[:, 0] = cp.arange(0, 5, dtype=cp.uint32)
    seq_ids[:, 2] = cp.asarray(host__seq_data)[:, 2]
    tensors['seq_ids'] = seq_ids

    memory = TensorMemory(count=5, tensors=tensors)

    msg = ControlMessage()
    msg.payload(meta)
    msg.tensors(memory)

    return msg


@pytest.mark.import_mod(os.path.join(TEST_DIRS.examples_dir, 'log_parsing', 'postprocessing.py'))
def test_log_parsing_post_processing_stage(config: Config,
                                           dataset_cudf: DatasetManager,
                                           import_mod: typing.List[types.ModuleType],
                                           bert_cased_vocab: str,
                                           model_config_file: str):
    postprocessing_mod = import_mod

    log_test_data_dir = os.path.join(TEST_DIRS.tests_data_dir, 'examples/log_parsing')

    stage = postprocessing_mod.LogParsingPostProcessingStage(config,
                                                             vocab_path=bert_cased_vocab,
                                                             model_config_path=model_config_file)

    post_proc_message = build_post_proc_message(dataset_cudf, log_test_data_dir)
    expected_df = dataset_cudf.pandas[os.path.join(log_test_data_dir, 'expected_out.csv')]

    out_meta = stage._postprocess(post_proc_message)

    assert isinstance(out_meta, MessageMeta)
    DatasetManager.assert_compare_df(out_meta.df, expected_df)


@pytest.mark.import_mod(os.path.join(TEST_DIRS.examples_dir, 'log_parsing', 'postprocessing.py'))
def test_undefined_variable_error(caplog: pytest.LogCaptureFixture,
                                  config: Config,
                                  dataset_cudf: DatasetManager,
                                  import_mod: typing.List[types.ModuleType],
                                  bert_cased_vocab: str,
                                  model_config_file: str):
    """
    Test for undefined variable error, which occurrs when the first token_id is unexpected resulting in the `new_label`
    and `new_confidence` variables being undefined.
    """
    postprocessing_mod = import_mod

    log_test_data_dir = os.path.join(TEST_DIRS.tests_data_dir, 'examples/log_parsing')

    stage = postprocessing_mod.LogParsingPostProcessingStage(config,
                                                             vocab_path=bert_cased_vocab,
                                                             model_config_path=model_config_file)

    post_proc_message = build_post_proc_message(dataset_cudf, log_test_data_dir)
    post_proc_message.tensors().get_tensor('input_ids')[0] = 27716.0

    expected_log_re = re.compile(r"^Ignoring unexecpected subword token:.*")

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        stage._postprocess(post_proc_message)

        logged_warning = False
        for rec in caplog.records:
            if rec.levelno == logging.WARNING and expected_log_re.match(rec.message) is not None:
                logged_warning = True
                break

            assert logged_warning, "Expected warning message not found in logs"
