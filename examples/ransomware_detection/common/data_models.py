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

import dataclasses
import typing

import pandas as pd


@dataclasses.dataclass
class FeatureConfig:
    """
    This dataclass holds a features creation configuration.
    """

    file_extns: typing.List[str]
    interested_plugins: typing.List[str]


@dataclasses.dataclass
class SnapshotData:
    """
    This dataclass holds appshield snapshot data.
    """

    snapshot_id: int
    data: typing.List[float]


@dataclasses.dataclass
class ProtectionData:
    """
    This dataclass contains protection data that is used to construct protection features.
    """

    commit_charges: pd.Series
    vads_protection_size: int
    vad_protection_size: int
    commit_charge_size: int
    protection_df_size: int
    protection_id: str
    vadinfo_df_size: int
    vadsinfo_size: int
    vadinfo_size: int
