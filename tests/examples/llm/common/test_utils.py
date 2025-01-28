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

import types

import pytest

try:
    import pymilvus
    from pymilvus.exceptions import DataTypeNotSupportException
except ImportError:
    pass


@pytest.fixture(name="pymilvus", scope='session', autouse=True)
def pymilvus_fixture(pymilvus: types.ModuleType):  # pylint: disable=redefined-outer-name
    """
    Fixture to ensure pymilvus is installed
    """
    yield pymilvus


def test_build_milvus_config_valid_schema(import_utils):
    resource_schema_config = {
        "schema_conf": {
            "schema_fields": [{
                "name": "field1", "dtype": "INT64"
            }, {
                "name": "field2", "dtype": "FLOAT"
            }]
        }
    }
    expected_dtype_map = {"field1": pymilvus.DataType.INT64, "field2": pymilvus.DataType.FLOAT}
    result = import_utils.build_milvus_config(resource_schema_config)
    for field in result["schema_conf"]["schema_fields"]:
        assert field["type"] == expected_dtype_map[field["name"]]


def test_build_milvus_config_invalid_dtype(import_utils):
    resource_schema_config = {"schema_conf": {"schema_fields": [{"name": "invalid_field", "dtype": "invalid_dtype"}]}}
    with pytest.raises(DataTypeNotSupportException):
        import_utils.build_milvus_config(resource_schema_config)


def test_build_milvus_config_empty_schema_fields(import_utils):
    resource_schema_config = {"schema_conf": {"schema_fields": []}}
    result = import_utils.build_milvus_config(resource_schema_config)
    assert result["schema_conf"]["schema_fields"] == []


def test_build_milvus_config_none_schema_config(import_utils):
    with pytest.raises(TypeError):
        import_utils.build_milvus_config(None)


def test_build_milvus_config_additional_field_properties(import_utils):
    with pytest.raises(DataTypeNotSupportException):
        resource_schema_config = {
            "schema_conf": {
                "schema_fields": [{
                    "name": "field1", "dtype": "int64", "extra_prop": "value"
                }]
            }
        }
        result = import_utils.build_milvus_config(resource_schema_config)
        assert "extra_prop" in result["schema_conf"]["schema_fields"][0]
        assert result["schema_conf"]["schema_fields"][0]["extra_prop"] == "value"
