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

import typing

import numpy as np
import pandas as pd
import pytest
import typing_utils

from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import TensorMemory
from morpheus.messages.message_meta import MessageMeta
from morpheus.stages.postprocess.add_classifications_stage import AddClassificationsStage
from morpheus.stages.postprocess.add_scores_stage import AddScoresStage


@pytest.fixture(name='config')
def fixture_config(config: Config):
    config.class_labels = ['frogs', 'lizards', 'toads']
    config.feature_length = 12
    yield config


def test_constructor(config: Config):
    stage = AddScoresStage(config)
    assert stage._class_labels == ['frogs', 'lizards', 'toads']
    assert stage._labels == ['frogs', 'lizards', 'toads']
    assert stage._idx2label == {0: 'frogs', 1: 'lizards', 2: 'toads'}
    assert stage.name == "add-scores"

    accepted_union = typing.Union[stage.accepted_types()]
    assert typing_utils.issubtype(ControlMessage, accepted_union)


def test_constructor_explicit_labels(config: Config):
    stage = AddScoresStage(config, labels=['lizards'], prefix='test_')
    assert stage._class_labels == ['frogs', 'lizards', 'toads']
    assert stage._labels == ['lizards']
    assert stage._idx2label == {1: 'test_lizards'}


def test_constructor_errors(config: Config):
    with pytest.raises(AssertionError):
        AddScoresStage(config, labels=['missing'])


@pytest.mark.cpu_mode
def test_add_labels_with_control_message():
    class_labels = {0: "frogs", 1: "lizards", 2: "toads"}

    df = pd.DataFrame([0, 1], columns=["dummy"])
    probs_array = np.array([[0.1, 0.5, 0.8], [0.2, 0.6, 0.9]])

    cm = ControlMessage()
    cm.payload(MessageMeta(df))
    cm.tensors(TensorMemory(count=2, tensors={"probs": probs_array}))

    labeled_cm = AddClassificationsStage._add_labels(cm, idx2label=class_labels, threshold=None)

    DatasetManager.assert_df_equal(labeled_cm.payload().get_data("frogs"), probs_array[:, 0])
    DatasetManager.assert_df_equal(labeled_cm.payload().get_data("lizards"), probs_array[:, 1])
    DatasetManager.assert_df_equal(labeled_cm.payload().get_data("toads"), probs_array[:, 2])

    # Too small of a probs array
    cm = ControlMessage()
    cm.payload(MessageMeta(df))
    cm.tensors(TensorMemory(count=2, tensors={"probs": probs_array[:, 0:-1]}))

    with pytest.raises(RuntimeError):
        AddClassificationsStage._add_labels(cm, idx2label=class_labels, threshold=None)
