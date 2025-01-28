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

from abc import ABC
from abc import abstractmethod
from typing import Union

import pandas as pd

import cudf


class RecordStorageInterface(ABC):

    def __init__(self, file_format: str):
        self._file_format = file_format
        self._backing_source = None
        self._owner = False

    @abstractmethod
    def store(self, data_source: Union[pd.DataFrame, cudf.DataFrame, str]) -> None:
        pass

    @abstractmethod
    def load(self) -> cudf.DataFrame:
        pass

    @abstractmethod
    def delete(self) -> None:
        pass

    @property
    @abstractmethod
    def backing_source(self) -> str:
        pass

    @property
    @abstractmethod
    def num_rows(self) -> int:
        pass

    @property
    @abstractmethod
    def owner(self) -> bool:
        pass
