# Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

import pytest


def test_is_valid_service_with_valid_name(import_vdb_update_utils_module):
    assert import_vdb_update_utils_module.is_valid_service(None, None, "milvus") == "milvus"


def test_is_valid_service_with_invalid_name(import_vdb_update_utils_module):
    with pytest.raises(ValueError):
        import_vdb_update_utils_module.is_valid_service(None, None, "invalid_service")


def test_is_valid_service_with_mixed_case(import_vdb_update_utils_module):
    assert import_vdb_update_utils_module.is_valid_service(None, None, "MilVuS") == "milvus"


def test_merge_dicts_non_overlapping(import_vdb_update_utils_module):
    file_config = {"key1": "value1"}
    cli_config = {"key2": "value2"}
    expected = {"key1": "value1", "key2": "value2"}
    assert import_vdb_update_utils_module.merge_dicts(file_config, cli_config) == expected


def test_merge_dicts_overlapping(import_vdb_update_utils_module):
    file_config = {"key1": "value1", "key2": "old_value"}
    cli_config = {"key2": "new_value"}
    expected = {"key1": "value1", "key2": "new_value"}
    assert import_vdb_update_utils_module.merge_dicts(file_config, cli_config) == expected


def test_merge_dicts_none_in_cli(import_vdb_update_utils_module):
    file_config = {"key1": "value1", "key2": "value2"}
    cli_config = {"key2": None}
    expected = {"key1": "value1", "key2": None}
    assert import_vdb_update_utils_module.merge_dicts(file_config, cli_config) == expected


def test_merge_dicts_empty(import_vdb_update_utils_module):
    file_config = {}
    cli_config = {"key1": "value1"}
    expected = {"key1": "value1"}
    assert import_vdb_update_utils_module.merge_dicts(file_config, cli_config) == expected
