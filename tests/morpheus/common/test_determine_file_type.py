# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

import pathlib

import pytest

from morpheus.common import FileTypes
from morpheus.common import determine_file_type


@pytest.mark.parametrize("use_pathlib", [False, True])
@pytest.mark.parametrize("ext, expected_result",
                         [("csv", FileTypes.CSV), ("json", FileTypes.JSON), ("jsonlines", FileTypes.JSON),
                          ("parquet", FileTypes.PARQUET)])
def test_determine_file_type(ext: str, expected_result: FileTypes, use_pathlib: bool):
    file_path = f"test.{ext}"
    if use_pathlib:
        file_path = pathlib.Path(file_path)

    assert determine_file_type(file_path) == expected_result
