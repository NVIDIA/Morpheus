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

from unittest import mock

import pytest

from _utils import make_mock_response
from morpheus.utils import http_utils


@pytest.mark.parametrize("url,expected",
                         [("nvidia.com/", "http://nvidia.com/"), ("https://nvidia.com", "https://nvidia.com"),
                          ("http://nvidia.com", "http://nvidia.com"),
                          ("localhost:8080/a/b/c?test=this", "http://localhost:8080/a/b/c?test=this"),
                          ("ftp://tester@nvidia.com/robots.txt", "ftp://tester@nvidia.com/robots.txt")])
def test_prepare_url(url: str, expected: str):
    assert http_utils.prepare_url(url) == expected


def test_prepare_url_error():
    with pytest.raises(ValueError):
        http_utils.prepare_url("")


@pytest.mark.parametrize("use_on_success_fn", [True, False])
@mock.patch("requests.Session")
@mock.patch("time.sleep")
def test_request_with_retry(mock_sleep: mock.MagicMock, mock_request_session: mock.MagicMock, use_on_success_fn: bool):
    mock_response = make_mock_response(mock_request_session)

    if use_on_success_fn:
        on_success_fn = mock.MagicMock()
        on_success_fn.return_value = {'on_success_fn': 'mock'}
    else:
        on_success_fn = None

    request_kwargs = {'method': 'GET', 'url': 'http://test.nvidia.com'}
    response = http_utils.request_with_retry(request_kwargs, on_success_fn=on_success_fn)

    if use_on_success_fn:
        assert response == (mock_request_session, {'on_success_fn': 'mock'})
    else:
        assert response == (mock_request_session, mock_response)

    mock_request_session.request.assert_called_once_with(method='GET', url='http://test.nvidia.com')
    mock_sleep.assert_not_called()

    if use_on_success_fn:
        on_success_fn.assert_called_once_with(mock_response)


@pytest.mark.parametrize("use_on_success_fn", [True, False])
@pytest.mark.parametrize("respect_retry_after_header", [True, False])
@mock.patch("requests.Session")
@mock.patch("time.sleep")
def test_request_with_retry_max_retries(mock_sleep: mock.MagicMock,
                                        mock_request_session: mock.MagicMock,
                                        respect_retry_after_header: bool,
                                        use_on_success_fn: bool):
    mock_response = make_mock_response(mock_request_session, status_code=500)
    mock_response.headers = {"Retry-After": "42"} if respect_retry_after_header else {}

    if use_on_success_fn:
        on_success_fn = mock.MagicMock()
    else:
        on_success_fn = None

    with pytest.raises(RuntimeError):
        http_utils.request_with_retry({
            'method': 'GET', 'url': 'http://test.nvidia.com'
        },
                                      max_retries=5,
                                      sleep_time=1,
                                      on_success_fn=on_success_fn)

    assert mock_request_session.request.call_count == 5
    assert mock_sleep.call_count == 4

    if respect_retry_after_header:
        mock_sleep.assert_has_calls([mock.call(42)] * 4)
    else:
        mock_sleep.assert_has_calls([
            mock.call(1),
            mock.call(2),
            mock.call(4),
            mock.call(8),
        ])

    if use_on_success_fn:
        on_success_fn.assert_not_called()
