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

# Original Source: https:#github.com/AlliedToasters/dfencoder
#
# Original License: BSD-3-Clause license, included below

# Copyright (c) 2019, Michael Klear.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.
#
#     * Neither the name of the dfencoder Developers nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import typing

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer


def ensure_float_type(x: typing.Union[torch.Tensor, np.ndarray]):
    """Ensure we are in the right floating point format. """
    if (isinstance(x, torch.Tensor)):
        result = x.to(dtype=torch.float32, copy=True)
    elif (isinstance(x, np.ndarray)):
        result = x.astype(float)
    else:
        raise ValueError(f"Unsupported type: {type(x)}")
    return result


class StandardScaler(object):
    """Impliments standard (mean/std) scaling."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: torch.Tensor):
        self.mean = x.mean().item()
        self.std = x.std().item()

        # Having a std == 0 (when all values are the same), breaks training. Just use 1.0 in this case
        if (self.std == 0):
            self.std = 1.0

    def transform(self, x: typing.Union[torch.Tensor, np.ndarray]):
        result = ensure_float_type(x)
        result -= self.mean
        result /= self.std
        return result

    def inverse_transform(self, x: torch.Tensor):
        result = ensure_float_type(x)
        result *= self.std
        result += self.mean
        return result

    def fit_transform(self, x: torch.Tensor):
        self.fit(x)
        return self.transform(x)


class ModifiedScaler(object):
    """Implements scaling using modified z score.
    Reference: https://www.ibm.com/docs/el/cognos-analytics/11.1.0?topic=terms-modified-z-score
    """
    MAD_SCALING_FACTOR = 1.486  # 1.486 * MAD approximately equals the standard deviation
    MEANAD_SCALING_FACTOR = 1.253314  # 1.253314 * MeanAD approximately equals the standard deviation

    def __init__(self):
        self.median: float = None
        self.mad: float = None  # median absolute deviation
        self.meanad: float = None  # mean absolute deviation

    def fit(self, x: torch.Tensor):
        med = x.median().item()
        self.median = med
        self.mad = (x - med).abs().median().item()
        self.meanad = (x - med).abs().mean().item()
        # Having a meanad == 0 (when all values are the same), breaks training. Just use 1.0 in this case
        if (self.meanad == 0):
            self.meanad = 1.0

    def transform(self, x: typing.Union[torch.Tensor, np.ndarray]):
        result = ensure_float_type(x)

        result -= self.median
        if self.mad == 0:
            result /= (self.MEANAD_SCALING_FACTOR * self.meanad)
        else:
            result /= (self.MAD_SCALING_FACTOR * self.mad)
        return result

    def inverse_transform(self, x: torch.Tensor):
        result = ensure_float_type(x)

        if self.mad == 0:
            result *= (self.MEANAD_SCALING_FACTOR * self.meanad)
        else:
            result *= (self.MAD_SCALING_FACTOR * self.mad)
        result += self.median
        return result

    def fit_transform(self, x: torch.Tensor):
        self.fit(x)
        return self.transform(x)


class GaussRankScaler(object):
    """
    So-called "Gauss Rank" scaling.
    Forces a transformation, uses bins to perform inverse mapping.

    Uses sklearn QuantileTransformer to work.
    """

    def __init__(self):
        self.transformer = QuantileTransformer(output_distribution='normal')

    def fit(self, x):
        x = x.reshape(-1, 1)
        self.transformer.fit(x)

    def transform(self, x):
        x = x.reshape(-1, 1)
        result = self.transformer.transform(x)
        return result.reshape(-1)

    def inverse_transform(self, x):
        x = x.reshape(-1, 1)
        result = self.transformer.inverse_transform(x)
        return result.reshape(-1)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class NullScaler(object):

    def __init__(self):
        pass

    def fit(self, x):
        pass

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x

    def fit_transform(self, x):
        return self.transform(x)
