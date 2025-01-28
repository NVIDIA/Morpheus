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

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from morpheus.models.dfencoder.ae_feature_selector import AutoencoderFeatureSelector


@pytest.fixture
def sample_data():
    """Fixture to provide the Iris dataset for testing."""
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    return data


def test_preprocess_data(sample_data):
    """Test the preprocess_data method."""
    selector = AutoencoderFeatureSelector(sample_data.to_dict(orient='records'))
    processed_data = selector.preprocess_data()
    assert isinstance(processed_data, np.ndarray)
    assert processed_data.shape[1] > 0  # Ensure columns exist after preprocessing


def test_remove_low_variance(sample_data):
    """Test removing low variance features."""
    sample_data['low_variance'] = 0
    selector = AutoencoderFeatureSelector(sample_data.to_dict(orient='records'), variance_threshold=0.1)
    reduced_data, mask = selector.remove_low_variance(sample_data.values)
    assert reduced_data.shape == (150,4)


def test_remove_high_correlation(sample_data):
    """Test removing highly correlated features."""
    sample_data['high_corr_1'] = sample_data['sepal length (cm)'] + 1
    sample_data['high_corr_2'] = 2 * sample_data['sepal length (cm)'] + 1
    selector = AutoencoderFeatureSelector(sample_data.to_dict(orient='records'), variance_threshold=0.1)
    reduced_data, mask = selector.remove_high_correlation(sample_data.values, threshold=0.99)
    assert reduced_data.shape == (150,4)
    assert mask == [4,5]


def test_select_features(sample_data):
    """Test the select_features method."""
    selector = AutoencoderFeatureSelector(sample_data.to_dict(orient='records'))
    raw_schema, preproc_schema = selector.select_features(k_min=3, k_max=5)
    assert isinstance(raw_schema, dict)
    assert isinstance(preproc_schema, dict)
