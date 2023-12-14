#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import MagicMock

import pytest
from requests.models import Response
from requests.sessions import Session

from morpheus.io.tao_client import TaoApiClient
from morpheus.io.tao_client import apikey_type_check
from morpheus.io.tao_client import generate_schema_url


def test_generate_schema_url():

    actual = generate_schema_url(url="localhost:32080", ssl=False)
    expected = "http://localhost:32080"

    assert actual == expected

    with pytest.raises(ValueError):
        generate_schema_url(url="http://localhost:32080", ssl=False)

    with pytest.raises(ValueError):
        generate_schema_url(url="https://localhost:32080", ssl=True)

    actual = generate_schema_url(url="localhost:32080", ssl=False)
    expected = "http://localhost:32080"
    assert actual == expected

    actual = generate_schema_url(url="localhost:32080", ssl=True)
    expected = "https://localhost:32080"
    assert actual == expected


def test_apikey_type_check():

    apikey_type_check("test_api_key")

    with pytest.raises(ValueError):
        apikey_type_check("")

    with pytest.raises(ValueError):
        apikey_type_check(123459)


def get_tao_client():
    mock_creds = {"user_id": "X20109876", "token": "TOkJJTkw6WkxKRDpNWk9ZOkRVN0o6"}
    tao_client = TaoApiClient("test_api_key", "localhost:32080")
    tao_client.authorize = MagicMock(return_value=mock_creds)

    return tao_client


def test_create_dataset_resource():
    tao_client = get_tao_client()

    mock_response = Response()
    mock_response.status_code = 201
    mock_response._content = b'{ "id" : "eyJzdWIiOiJwOTltOTh0NzBzdDFsa3Zx", "created_on": "2023-01-17T15:35:08.014463"}'

    tao_client.session.post = MagicMock(return_value=mock_response)

    ds_type = "object_detection"
    ds_format = "kitti"

    data = {"type": ds_type, "format": ds_format}

    resource_id = tao_client.create_resource("dataset", data)

    with pytest.raises(ValueError):
        tao_client.create_resource("test", data=data)

    assert resource_id == "eyJzdWIiOiJwOTltOTh0NzBzdDFsa3Zx"

    mock_response2 = Response()
    mock_response2.status_code = 400
    mock_response2._content = b'''{ "id" : "eyJzdWIiOiJwOTltOTh0NzBzdDFsa3Zx",
    "created_on": "2023-01-17T15:35:08.014463"}'''

    tao_client.session.post = MagicMock(return_value=mock_response2)

    with pytest.raises(Exception):
        tao_client.create_resource("dataset", data=data)


def test_create_model_resource():
    tao_client = get_tao_client()

    mock_response = Response()
    mock_response.status_code = 201
    mock_response._content = b'{ "id" : "eyJzdWIiOiJwOTltOTh0NzBzdDFsa3Zx", "created_on": "2023-01-17T15:35:08.014463"}'

    tao_client.session.post = MagicMock(return_value=mock_response)

    network_arch = "detectnet_v2"
    encode_key = "tlt_encode"
    data = {"network_arch": network_arch, "encryption_key": encode_key, "description": "My model"}

    resource_id = tao_client.create_resource("model", data)

    assert resource_id == "eyJzdWIiOiJwOTltOTh0NzBzdDFsa3Zx"

    with pytest.raises(ValueError):
        tao_client.create_resource("random_kind", data=data)


def test_partial_update_resource():

    tao_client = get_tao_client()

    mock_response = Response()
    mock_response.status_code = 200
    mock_response._content = b'{ "id" : "eyJzdWIiOiJwOTltOTh0NzBzdDFsa3Zx", "created_on": "2023-01-17T15:35:08.014463"}'

    tao_client.session.patch = MagicMock(return_value=mock_response)

    data = {"name": "Train dataset", "description": "My train dataset with kitti"}

    resp_json = tao_client.partial_update_resource("dataset", resource_id="eyJzdWIiOiJwOTltOTh0NzBzdDFsa3Zx", data=data)

    assert resp_json.get("id") == "eyJzdWIiOiJwOTltOTh0NzBzdDFsa3Zx"

    mock_response.status_code = 401
    tao_client.session.patch = MagicMock(return_value=mock_response)

    with pytest.raises(Exception):
        tao_client.create_resource("dataset", data=data)


def test_update_resource():

    tao_client = get_tao_client()

    mock_response = Response()
    mock_response.status_code = 200
    mock_response._content = b'{ "id" : "eyJzdWIiOiJwOTltOTh0NzBzdDFsa3Zx", "created_on": "2023-01-17T15:35:08.014463"}'

    tao_client.session.put = MagicMock(return_value=mock_response)

    data = {"name": "Train dataset", "description": "My train dataset with kitti"}

    resp_json = tao_client.update_resource("dataset", resource_id="eyJzdWIiOiJwOTltOTh0NzBzdDFsa3Zx", data=data)

    assert resp_json.get("id") == "eyJzdWIiOiJwOTltOTh0NzBzdDFsa3Zx"
    assert isinstance(resp_json, dict)


def test_get_specs_schema():
    tao_client = get_tao_client()

    mock_response = Response()
    mock_response.status_code = 200

    mock_response._content = b'{ "id" : "eyJzdWIiOiJwOTltOTh0NzBzdDFsa3Zx", "created_on": "2023-01-17T15:35:08.014463"}'

    resource_id = "eyJzdWIiOiJwOTltOTh0NzBzdDFsa3Zx"

    tao_client.session.get = MagicMock(return_value=mock_response)

    resp_json = tao_client.get_specs_schema("dataset", "convert", resource_id=resource_id)

    with pytest.raises(ValueError):
        tao_client.get_specs_schema("dataset", "tmp_convert", resource_id=resource_id)

    assert resp_json.get("id") == "eyJzdWIiOiJwOTltOTh0NzBzdDFsa3Zx"
    assert isinstance(resp_json, dict)


def test_close():
    tao_client = get_tao_client()

    session = tao_client.session
    assert isinstance(session, Session)

    tao_client.close()
    assert tao_client.session is None


def test_upload_resource(tmpdir):
    input_data = tmpdir.join("input_dataset.txt")

    with open(input_data, 'w') as fh:
        fh.write("This is a training data.")

    tao_client = get_tao_client()

    mock_response = Response()
    mock_response.status_code = 201
    mock_response._content = b'{ "id" : "eyJzdWIiOiJwOTltOTh0NzBzdDFsa3Zx", "created_on": "2023-01-17T15:35:08.014463"}'

    resource_id = "eyJzdWIiOiJwOTltOTh0NzBzdDFsa3Zx"

    tao_client.session.post = MagicMock(return_value=mock_response)

    resp_json = tao_client.upload_resource("dataset", resource_path=input_data, resource_id=resource_id)

    assert resp_json.get("id") == "eyJzdWIiOiJwOTltOTh0NzBzdDFsa3Zx"
    assert isinstance(resp_json, dict)


def test_download_resource(tmpdir):
    tao_client = get_tao_client()

    mock_response = Response()
    mock_response.status_code = 200
    mock_response._content = b'''{ "id" : "eyJzdWIiOiJwOTltOTh0NzBzdDFsa3Zx", "parent_id" : "None",
    "status": "Pending", "created_on": "2023-01-17T15:35:08.014463"}'''

    resource_id = "eyJzdWIiOiJwOTltOTh0NzBzdDFsa3Zx"

    tao_client.session.get = MagicMock(return_value=mock_response)

    resp_json = tao_client.download_resource("dataset",
                                             resource_id=resource_id,
                                             job_id="test_235678",
                                             output_dir=tmpdir)

    assert resp_json is None
