# Copyright (c) 2022-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Type aliases used throughout the codebase."""

import typing

if typing.TYPE_CHECKING:
    import cupy
    import numpy
    import pandas

    import cudf

DataFrameModule = typing.Literal["cudf", "pandas"]
"""Valid DataFrame modules."""

DataFrameType = typing.Union["pandas.DataFrame", "cudf.DataFrame"]
"""Alias for pandas and cuDF DataFrame types."""

SeriesType = typing.Union["pandas.Series", "cudf.Series"]
"""Alias for pandas and cuDF Series types."""

NDArrayType = typing.Union["numpy.ndarray", "cupy.ndarray"]
"""Alias for NumPy and CuPy ndarray types."""

# Intentionally using `typing.Dict` instead of `dict` to avoid a Sphinx build error.
# https://github.com/nv-morpheus/Morpheus/issues/1956
TensorMapType = typing.Dict[str, NDArrayType]
"""Alias for a dictionary of tensor names to tensors represented as either a NumPy or CuPy ndarray."""
