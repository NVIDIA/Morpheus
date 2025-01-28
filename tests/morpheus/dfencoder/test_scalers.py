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

import warnings

import numpy as np
import pytest
import torch

from morpheus.models.dfencoder import scalers

# Pylint doesn't understand how pytest fixtures work and flags fixture uasage as a redefinition of the symbol in the
# outer scope.
# pylint: disable=redefined-outer-name


@pytest.fixture(name="fit_tensor", scope="function")
def fit_tensor_fixture():
    yield torch.tensor([4.4, 5.3, 6.5], dtype=torch.float32)


@pytest.fixture(name="tensor", scope="function")
def tensor_fixture():
    yield torch.tensor([7.4, 8.3, 9.5], dtype=torch.float32)


@pytest.fixture(name="standard_scaler", scope="function")
def standard_scaler_fixture(fit_tensor):
    scaler = scalers.StandardScaler()
    scaler.fit(fit_tensor)
    yield scaler


@pytest.fixture(name="modified_scaler", scope="function")
def modified_scaler_fixture(fit_tensor):
    scaler = scalers.ModifiedScaler()
    scaler.fit(fit_tensor)
    yield scaler


@pytest.fixture(name="gauss_rank_scaler", scope="function")
def gauss_rank_scaler_fixture(fit_tensor):
    scaler = scalers.GaussRankScaler()

    with warnings.catch_warnings():
        # This warning is triggered by the abnormally small tensor size used in this test
        warnings.filterwarnings("ignore",
                                message=r"n_quantiles \(1000\) is greater than the total number of samples \(3\).*",
                                category=UserWarning)
        scaler.fit(fit_tensor)
        yield scaler


def test_ensure_float_type():
    result = scalers.ensure_float_type(np.ones(10, np.int32))
    assert result.dtype == np.float64

    result = scalers.ensure_float_type(torch.ones(10, dtype=torch.int32))
    assert result.dtype == torch.float32

    with pytest.raises(ValueError):
        scalers.ensure_float_type([1, 2, 3])


def test_standard_scaler_fit(standard_scaler):
    assert round(standard_scaler.mean, 2) == 5.4
    assert round(standard_scaler.std, 2) == 1.05

    # Test corner case where all values are the same
    standard_scaler.fit(torch.ones(5, dtype=torch.float32))
    assert standard_scaler.mean == 1
    assert standard_scaler.std == 1.0


def test_standard_scaler_transform(standard_scaler, tensor):
    results = standard_scaler.transform(tensor)
    expected = torch.tensor([1.9, 2.75, 3.89])
    assert torch.equal(torch.round(results, decimals=2), expected), f"{results} != {expected}"


def test_standard_scaler_inverse_transform(standard_scaler, tensor):
    results = standard_scaler.inverse_transform(tensor)
    expected = torch.tensor([13.2, 14.14, 15.41])
    assert torch.equal(torch.round(results, decimals=2), expected), f"{results} != {expected}"


def test_standard_scaler_fit_transform(standard_scaler, tensor):
    results = standard_scaler.fit_transform(tensor)
    expected = torch.tensor([-0.95, -0.09, 1.04])
    assert torch.equal(torch.round(results, decimals=2), expected), f"{results} != {expected}"


def test_modified_scaler_fit(modified_scaler):
    assert round(modified_scaler.median, 2) == 5.3
    assert round(modified_scaler.mad, 2) == 0.9
    assert round(modified_scaler.meanad, 2) == 0.7

    # Test corner case where all values are the same
    modified_scaler.fit(torch.ones(5, dtype=torch.float32))
    assert modified_scaler.meanad == 1.0


def test_modified_scaler_transform(modified_scaler, tensor):
    results = modified_scaler.transform(tensor)
    expected = torch.tensor([1.57, 2.24, 3.14])
    assert torch.equal(torch.round(results, decimals=2), expected), f"{results} != {expected}"

    # Test alternate path where median absolute deviation is 1
    modified_scaler.fit(torch.tensor([3.0, 4.0, 4.0, 5.0]))
    results = modified_scaler.transform(tensor)
    expected = torch.tensor([5.43, 6.86, 8.78])
    assert torch.equal(torch.round(results, decimals=2), expected), f"{results} != {expected}"


def test_modified_scaler_inverse_transform(modified_scaler, tensor):
    results = modified_scaler.inverse_transform(tensor)
    expected = torch.tensor([15.2, 16.40, 18.01])
    assert torch.equal(torch.round(results, decimals=2), expected), f"{results} != {expected}"

    # Test alternate path where median absolute deviation is 1
    modified_scaler.fit(torch.tensor([3.0, 4.0, 4.0, 5.0]))
    results = modified_scaler.inverse_transform(tensor)
    expected = torch.tensor([8.64, 9.2, 9.95])
    assert torch.equal(torch.round(results, decimals=2), expected), f"{results} != {expected}"


def test_modified_scaler_fit_transform(modified_scaler, tensor):
    results = modified_scaler.fit_transform(tensor)
    expected = torch.tensor([-0.67, 0.0, 0.9])
    assert torch.equal(torch.round(results, decimals=2), expected), f"{results} != {expected}"


def test_gauss_rank_scaler_transform(gauss_rank_scaler, tensor):
    results = gauss_rank_scaler.transform(tensor)
    expected = np.array([5.2, 5.2, 5.2])
    assert results.round(2).tolist() == expected.tolist(), f"{results} != {expected}"


def test_gauss_rank_scaler_inverse_transform(gauss_rank_scaler, tensor):
    results = gauss_rank_scaler.inverse_transform(tensor)
    expected = np.array([6.5, 6.5, 6.5])
    assert results.round(2).tolist() == expected.tolist(), f"{results} != {expected}"


def test_gauss_rank_scaler_fit_transform(gauss_rank_scaler, tensor):
    results = gauss_rank_scaler.fit_transform(tensor)
    expected = np.array([-5.2, 0.0, 5.2])
    assert results.round(2).tolist() == expected.tolist(), f"{results} != {expected}"


def test_null_scaler(tensor):
    orig = tensor.to(dtype=torch.float32, copy=True)
    null_scaler = scalers.NullScaler()
    null_scaler.fit(tensor)

    # Verify it does nothing
    assert null_scaler.transform(tensor) is tensor
    assert null_scaler.inverse_transform(tensor) is tensor
    assert null_scaler.fit_transform(tensor) is tensor

    # After all that the values should be the same
    assert torch.equal(tensor, orig), f"{tensor} != {orig}"
