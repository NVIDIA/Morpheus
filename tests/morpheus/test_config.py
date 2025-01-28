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

import json
import logging
import os
from dataclasses import FrozenInstanceError
from unittest import mock

import pytest

import morpheus
import morpheus.config
from _utils import assert_path_exists

# Using morpheus.config to distinguish between the config package and the pytest fixture of the same name


@mock.patch('docker.from_env')
def test_auto_determine_bootstrap(mock_docker_from_env):
    mock_net = mock.MagicMock()
    mock_net.get.return_value = mock_net
    mock_net.attrs = {'IPAM': {'Config': [{'Gateway': 'test_bridge_ip'}]}}

    # create mock containers
    mc1 = mock.MagicMock()
    mc1.ports = {'9092/tcp': [{'HostIp': 'test_host1', 'HostPort': '47'}]}

    mc2 = mock.MagicMock()
    mc2.ports = {}

    mc3 = mock.MagicMock()
    mc3.ports = {'9092/tcp': [{'HostIp': 'test_host2', 'HostPort': '42'}]}

    mock_net.containers = [mc1, mc2, mc3]

    mock_docker_client = mock.MagicMock()
    mock_docker_client.networks = mock_net
    mock_docker_from_env.return_value = mock_docker_client

    bootstrap_servers = morpheus.config.auto_determine_bootstrap()
    assert bootstrap_servers == "test_bridge_ip:47,test_bridge_ip:42"


def test_config_base(config):
    assert isinstance(config, morpheus.config.ConfigBase)


def test_config_onnx_to_trt():
    config_onnx_trt = morpheus.config.ConfigOnnxToTRT(input_model='frogs',
                                                      output_model='toads',
                                                      batches=[(1, 2), (3, 4)],
                                                      seq_length=100,
                                                      max_workspace_size=512)

    assert isinstance(config_onnx_trt, morpheus.config.ConfigBase)
    assert isinstance(config_onnx_trt, morpheus.config.ConfigOnnxToTRT)
    assert config_onnx_trt.input_model == 'frogs'
    assert config_onnx_trt.output_model == 'toads'
    assert config_onnx_trt.batches == [(1, 2), (3, 4)]
    assert config_onnx_trt.seq_length == 100
    assert config_onnx_trt.max_workspace_size == 512


def test_auto_encoder():
    config_ae = morpheus.config.ConfigAutoEncoder(feature_columns=['a', 'b', 'c', 'def'],
                                                  userid_column_name='def',
                                                  userid_filter='testuser')

    assert isinstance(config_ae, morpheus.config.ConfigBase)
    assert isinstance(config_ae, morpheus.config.ConfigAutoEncoder)
    assert config_ae.feature_columns == ['a', 'b', 'c', 'def']
    assert config_ae.userid_column_name == 'def'
    assert config_ae.userid_filter == 'testuser'


def test_pipeline_modes():
    expected = {"OTHER", "NLP", "FIL"}
    entries = set(pm.name for pm in morpheus.config.PipelineModes)
    assert entries.issuperset(expected)


def test_config_save(tmp_path):
    filename = os.path.join(tmp_path, 'config.json')

    config = morpheus.config.Config()
    config.save(filename)

    assert_path_exists(filename)
    with open(filename, encoding='UTF-8') as fh:
        assert isinstance(json.load(fh), dict)


def test_to_string(config):
    conf_str = config.to_string()
    assert isinstance(conf_str, str)
    assert isinstance(json.loads(conf_str), dict)


def test_frozen(config: morpheus.config.Config):
    assert not config.frozen

    # Ensure that it is safe to call freeze() multiple times
    for _ in range(2):
        config.freeze()
        assert config.frozen


@pytest.mark.parametrize('use_attr', [False, True])
def test_frozen_immutible(config: morpheus.config.Config, use_attr: bool):
    """
    Test for the freeze functionality.

    There are currently two ways to bypass the freeze functionality:
    1. By accessing the __dict__ attribute of the Config object.
    2. Modifying any of the mutable objects in the Config object (ex: `config.class_labels.append('new_label')`).
    """
    assert not config.frozen

    # ensure that we can set some attributes
    config.feature_length = 45

    # freeze the config, freezing the config via the attribute or method should have the same effect, the only
    # difference is that it is safe to call freeze() multiple times, while setting the attribute will raise an exception
    # just like attempting to set any other attribute on a frozen object
    if use_attr:
        config.frozen = True
    else:
        config.freeze()

    assert config.frozen

    with pytest.raises(FrozenInstanceError):
        config.feature_length = 100

    # ensure setattr also raises an exception
    with pytest.raises(FrozenInstanceError):
        setattr(config, 'feature_length', 100)

    # ensure the config still has the original value
    assert config.feature_length == 45


def test_warning_model_batch_size_less_than_pipeline_batch_size(caplog: pytest.LogCaptureFixture):
    config = morpheus.config.Config()
    config.pipeline_batch_size = 256
    with caplog.at_level(logging.WARNING):
        config.model_max_batch_size = 257
        config.freeze()
        assert len(caplog.records) == 1
        import re
        assert re.match(".*pipeline_batch_size < model_max_batch_size.*", caplog.records[0].message) is not None


def test_warning_pipeline_batch_size_less_than_model_batch_size(caplog: pytest.LogCaptureFixture):
    config = morpheus.config.Config()
    config.model_max_batch_size = 8
    with caplog.at_level(logging.WARNING):
        config.pipeline_batch_size = 7
        config.freeze()
        assert len(caplog.records) == 1
        import re
        assert re.match(".*pipeline_batch_size < model_max_batch_size.*", caplog.records[0].message) is not None
