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

import pandas as pd
import pytest

from morpheus.models.dfencoder.dataframe import EncoderDataFrame


def test_constructor():
    df = EncoderDataFrame([[1, 2, 3]])
    assert isinstance(df, pd.DataFrame)
    assert df.values.tolist() == [[1, 2, 3]]


@pytest.mark.usefixtures("manual_seed")
def test_swap():
    values = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

    # Docstrings imply the swaped values are returned as a copy, however the swaps are performed in-place,
    # then returned as a copy
    df = EncoderDataFrame(values)
    swapped = df.swap()
    expected = [[1, 2, 3], [4, 5, 6], [7, 11, 9], [10, 11, 3]]
    assert swapped.values.tolist() == expected

    df = EncoderDataFrame(values)
    swapped = df.swap(likelihood=0)
    assert swapped.values.tolist() == values
