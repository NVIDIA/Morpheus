# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import typing

import pytest
import typing_utils

from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage


@pytest.fixture(name='config')
def fixture_config(config: Config):
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
    config.edge_buffer_size = 4
    config.feature_length = 256
    config.mode = "NLP"
    config.model_max_batch_size = 32
    config.num_threads = 1
    config.pipeline_batch_size = 64
    yield config


def test_constructor(config: Config):
    stage = PreprocessNLPStage(config)
    assert stage.name == "preprocess-nlp"
    assert stage._column == "data"
    assert stage._seq_length == 256
    assert stage._vocab_hash_file.endswith("data/bert-base-cased-hash.txt")
    assert stage._truncation is False
    assert stage._do_lower_case is False
    assert stage._add_special_tokens is False

    accepted_union = typing.Union[stage.accepted_types()]
    assert typing_utils.issubtype(ControlMessage, accepted_union)
