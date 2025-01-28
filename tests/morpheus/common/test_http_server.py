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

import os
import time
import typing
from http import HTTPStatus
from unittest import mock

import pytest
import requests

from _utils import make_url
from morpheus.common import HttpEndpoint
from morpheus.common import HttpServer
from morpheus.utils.http_utils import MimeTypes


def make_parse_fn(status: HTTPStatus = HTTPStatus.OK,
                  content_type: str = "unit/test",
                  content: str = 'TEST OK',
                  on_complete_cb: typing.Callable = None) -> mock.MagicMock:
    mock_parse_fn = mock.MagicMock()
    mock_parse_fn.return_value = (status.value, content_type, content, on_complete_cb)
    return mock_parse_fn


@pytest.mark.slow
@pytest.mark.parametrize("endpoints", [("/t1", "/t2", "/t3"), ("test/", "123/", "a1d/"), ("/a", "/a/b", "/a/b/c/d")])
@pytest.mark.parametrize("port", [9090])
@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("include_headers", [True, False], ids=["with_headers", "without_headers"])
@pytest.mark.parametrize("use_callback", [True, False], ids=["with_callback", "without_callback"])
@pytest.mark.parametrize("use_context_mgr", [True, False], ids=["with_context_mgr", "without_context_mgr"])
@pytest.mark.parametrize("num_threads", [1, min(8, len(os.sched_getaffinity(0)))])
@pytest.mark.parametrize("status,content_type,content",
                         [(HTTPStatus.OK, MimeTypes.TEXT.value, "OK"),
                          (HTTPStatus.OK, MimeTypes.JSON.value, '{"test": "OK"}'),
                          (HTTPStatus.INTERNAL_SERVER_ERROR, MimeTypes.TEXT.value, "Unexpected error")])
def test_simple_request(port: int,
                        endpoints: typing.Tuple[str, str, str],
                        method: str,
                        status: HTTPStatus,
                        content_type: str,
                        content: str,
                        include_headers: bool,
                        use_callback: bool,
                        use_context_mgr: bool,
                        num_threads: int):
    if use_callback:
        callback_fn = mock.MagicMock()
    else:
        callback_fn = None

    parse_fn = make_parse_fn(status=status, content_type=content_type, content=content, on_complete_cb=callback_fn)

    if method == "GET":
        payload = ''
    else:
        if content_type == MimeTypes.JSON.value:
            payload = '{"test": "this"}'
        else:
            payload = "test"

    server = None

    def check_server(url: str, endpoint: str) -> None:
        assert server.is_running()

        response = requests.request(method=method, url=url, data=payload, timeout=5.0, headers={"unit": "test"})

        assert response.status_code == status.value
        assert response.headers["Content-Type"] == content_type
        assert response.text == content

        if include_headers:
            expected_endpoint = endpoint
            if not expected_endpoint.startswith("/"):
                expected_endpoint = f"/{expected_endpoint}"

            # Subset of headers that we want to check for
            expected_headers = {
                'Host': f'127.0.0.1:{port}',
                'endpoint': expected_endpoint,
                'method': method,
                'remote_address': '127.0.0.1',
                'unit': 'test'
            }
            parse_fn.assert_called_once()
            assert parse_fn.call_args[0][0] == payload

            actual_headers = parse_fn.call_args[0][1]
            for (key, value) in expected_headers.items():
                assert actual_headers[key] == value

        else:
            parse_fn.assert_called_once_with(payload)

        parse_fn.reset_mock()

        if use_callback:
            # Since the callback is executed asynchronously, we don't know when it will be called.
            expected_call = mock.call(False, "")

            max_retries = 300
            attempt = 0
            while expected_call not in callback_fn.mock_calls and attempt < max_retries:
                attempt += 1
                time.sleep(0.1)

            callback_fn.assert_called_once_with(False, "")
            callback_fn.reset_mock()

    urls = []
    http_endpoints = []

    for endpoint in endpoints:
        urls.append(make_url(port, endpoint))
        http_endpoints.append(
            HttpEndpoint(py_parse_fn=parse_fn, url=endpoint, method=method, include_headers=include_headers))

    if use_context_mgr:
        with HttpServer(endpoints=http_endpoints, port=port, num_threads=num_threads) as server:
            assert server.is_running()
            for (i, url) in enumerate(urls):
                check_server(url, endpoints[i])

    else:
        server = HttpServer(endpoints=http_endpoints, port=port, num_threads=num_threads)
        assert not server.is_running()
        server.start()

        for (i, url) in enumerate(urls):
            check_server(url, endpoints[i])

        server.stop()

    assert not server.is_running()


@pytest.mark.parametrize("endpoint", ["/test"])
def test_constructor_errors(endpoint: str):
    with pytest.raises(RuntimeError):
        HttpEndpoint(py_parse_fn=make_parse_fn(), url=endpoint, method="UNSUPPORTED")

    http_endpoint = HttpEndpoint(py_parse_fn=make_parse_fn(), url=endpoint, method="GET")
    with pytest.raises(RuntimeError):
        HttpServer(endpoints=[http_endpoint], num_threads=0)
