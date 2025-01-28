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
"""
Morpheus loader definitions, each loader is automatically registered when imported and will be available for use.
"""

from morpheus.loaders import file_to_df_loader
from morpheus.loaders import fsspec_loader
from morpheus.loaders import sql_loader

__all__ = ["file_to_df_loader", "fsspec_loader", "sql_loader"]
