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

import cupy as cp
import numpy as np
import pytest

import cudf

from _utils import TEST_DIRS
from _utils.dataset_manager import DatasetManager
from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta


def check_inf_message(msg: ControlMessage,
                      expected_mess_count: int,
                      expected_count: int,
                      expected_feature_length: int,
                      expected_flow_ids: cudf.Series,
                      expected_rollup_time: str,
                      expected_input__0: cp.ndarray):
    assert isinstance(msg, ControlMessage)
    assert msg.payload().count == expected_mess_count
    assert msg.tensor_count() == expected_count

    df = msg.payload().get_data()
    assert 'flow_id' in df
    assert 'rollup_time' in df

    assert (df.flow_id == expected_flow_ids).all()
    assert (df.rollup_time == expected_rollup_time).all()

    assert msg.tensors().has_tensor('input__0')
    assert msg.tensors().has_tensor('seq_ids')

    input__0 = msg.tensors().get_tensor('input__0')
    assert input__0.shape == (expected_count, expected_feature_length)
    assert input__0.dtype == cp.float32
    assert input__0.strides == (expected_feature_length * 4, 4)
    assert (input__0 == expected_input__0).all()

    seq_ids = msg.tensors().get_tensor('seq_ids')
    assert seq_ids.shape == (expected_count, 3)
    assert (seq_ids[:, 0] == cp.arange(0, expected_mess_count, dtype=cp.uint32)).all()
    assert (seq_ids[:, 1] == 0).all()
    assert (seq_ids[:, 2] == expected_feature_length - 1).all()


@pytest.mark.import_mod([os.path.join(TEST_DIRS.examples_dir, 'abp_pcap_detection/abp_pcap_preprocessing.py')])
def test_abp_pcap_preprocessing(config: Config, dataset_cudf: DatasetManager,
                                import_mod: typing.List[types.ModuleType]):
    # Setup the config
    config.mode = PipelineModes.FIL
    config.feature_length = 13

    abp_pcap_preprocessing = import_mod[0]

    # Get our input data, should contain the first 20 lines of the production data
    input_file = os.path.join(TEST_DIRS.tests_data_dir, 'examples/abp_pcap_detection/abp_pcap.jsonlines')
    input_df = dataset_cudf.get_df(input_file, no_cache=True, filter_nulls=False)

    expected_flow_ids = input_df.src_ip + ":" + input_df.src_port + "=" + input_df.dest_ip + ":" + input_df.dest_port
    expected_input__0 = cp.asarray(np.loadtxt(os.path.join(TEST_DIRS.tests_data_dir,
                                                           'examples/abp_pcap_detection/abp_pcap_expected_input_0.csv'),
                                              delimiter=",",
                                              skiprows=0,
                                              dtype=np.float32),
                                   order='C')

    assert len(input_df) == 20

    meta = MessageMeta(input_df[0:10])
    cm = ControlMessage()
    cm.payload(meta)

    stage = abp_pcap_preprocessing.AbpPcapPreprocessingStage(config)
    assert stage.get_needed_columns() == {'flow_id': TypeId.STRING, 'rollup_time': TypeId.STRING}

    inf = stage.pre_process_batch(cm, config.feature_length, stage.features, stage.req_cols)
    check_inf_message(inf,
                      expected_mess_count=10,
                      expected_count=10,
                      expected_feature_length=config.feature_length,
                      expected_flow_ids=expected_flow_ids[0:10],
                      expected_rollup_time='2021-04-07 15:55',
                      expected_input__0=expected_input__0[0:10])
