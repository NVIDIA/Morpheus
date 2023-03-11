#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import torch

from morpheus.models.dfencoder import autoencoder


def test_ohe():
    tensor = torch.tensor(range(4), dtype=torch.int64)
    results = autoencoder.ohe(tensor, 4, device="cpu")
    expected = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    assert results.device.type == "cpu"
    assert torch.equal(results, expected), f"{results} != {expected}"

    results = autoencoder.ohe(tensor.to("cuda", copy=True), 4, device="cuda")
    assert results.device.type == "cuda"
    assert torch.equal(results, expected.to("cuda", copy=True)), f"{results} != {expected}"


def test_compute_embedding_size():
    for (input, expected) in [(0, 0), (5, 4), (20, 9), (40000, 600)]:
        assert autoencoder.compute_embedding_size(input) == expected


def test_complete_layer_constructor():
    cc = autoencoder.CompleteLayer(4, 5)
    assert len(cc.layers) == 1
    assert isinstance(cc.layers[0], torch.nn.Linear)
    assert cc.layers[0].in_features == 4
    assert cc.layers[0].out_features == 5

    cc = autoencoder.CompleteLayer(4, 5, activation='tanh')
    assert len(cc.layers) == 2
    assert cc.layers[1] is torch.tanh

    cc = autoencoder.CompleteLayer(4, 5, dropout=0.2)
    assert len(cc.layers) == 2
    assert isinstance(cc.layers[1], torch.nn.Dropout)
    assert cc.layers[1].p == 0.2

    cc = autoencoder.CompleteLayer(6, 11, activation='sigmoid', dropout=0.3)
    assert len(cc.layers) == 3
    assert isinstance(cc.layers[0], torch.nn.Linear)
    assert cc.layers[0].in_features == 6
    assert cc.layers[0].out_features == 11
    assert cc.layers[1] is torch.sigmoid
    assert isinstance(cc.layers[2], torch.nn.Dropout)
    assert cc.layers[2].p == 0.3

    # Test for bad activation, this really does raise the base Exception class.
    with pytest.raises(Exception):
        autoencoder.CompleteLayer(4, 5, activation="does_not_exist")


def test_complete_layer_interpret_activation():
    cc = autoencoder.CompleteLayer(4, 5)
    assert cc.interpret_activation('elu') is torch.nn.functional.elu

    # Test for bad activation, this really does raise the base Exception class.
    with pytest.raises(Exception):
        cc.interpret_activation()

    with pytest.raises(Exception):
        cc.interpret_activation("does_not_exist")

    cc = autoencoder.CompleteLayer(6, 11, activation='sigmoid')
    cc.interpret_activation() is torch.sigmoid


@pytest.mark.usefixtures("manual_seed")
def test_complete_layer_forward():
    cc = autoencoder.CompleteLayer(3, 5, activation='tanh', dropout=0.2)
    t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.float32)
    results = cc.forward(t)
    expected = torch.tensor([[0.9029, 0.9877, 0.0000, 0.0000, 1.1453], [1.2464, 1.2372, 1.2485, 1.0397, 1.2490],
                             [1.2500, 1.2494, 1.2500, 1.1771, 1.2500], [1.2500, 0.0000, 1.2500, 1.2257, 1.2500]],
                            dtype=torch.float32)

    assert torch.equal(torch.round(results, decimals=4), expected), f"{results} != {expected}"
