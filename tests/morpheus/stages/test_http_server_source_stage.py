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

import queue
import threading
import time
import typing
from collections import deque
from http import HTTPStatus
from io import StringIO
from unittest import mock

import pandas as pd
import pytest
import requests
import requests.adapters
from urllib3.util.retry import Retry

from _utils import make_url
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.io.serializers import df_to_stream_json
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.configurable_output_source import SupportedMessageTypes
from morpheus.stages.input.http_server_source_stage import HttpServerSourceStage
from morpheus.utils.http_utils import HTTPMethod
from morpheus.utils.http_utils import MimeTypes


class GetNext(threading.Thread):

    def __init__(self, msg_queue: queue.Queue, generator: typing.Iterator[ControlMessage | MessageMeta]):
        threading.Thread.__init__(self)
        self._generator = generator
        self._msg_queue = msg_queue
        self._exception = None

    def run(self):
        try:
            msg = next(self._generator)
            self._msg_queue.put_nowait(msg)
        except Exception as e:
            print(f"Exception in GetNext thread: {e}")
            self._exception = e

    def join(self, timeout=None):
        super().join(timeout=timeout)
        if self._exception:
            raise self._exception


@pytest.mark.slow
@pytest.mark.cpu_mode
@pytest.mark.parametrize("message_type, task_type, task_payload",
                         [(SupportedMessageTypes.MESSAGE_META, None, None),
                          (SupportedMessageTypes.CONTROL_MESSAGE, None, None),
                          (SupportedMessageTypes.CONTROL_MESSAGE, "test", {
                              "pay": "load"
                          })],
                         ids=["message_meta", "control_message_no_task", "control_message_with_task"])
@pytest.mark.parametrize("lines", [False, True], ids=["json", "lines"])
@pytest.mark.parametrize("use_payload_to_df_fn", [False, True], ids=["no_payload_to_df_fn", "payload_to_df_fn"])
def test_generate_frames(config: Config,
                         mock_subscription: mock.MagicMock,
                         dataset_pandas: DatasetManager,
                         lines: bool,
                         use_payload_to_df_fn: bool,
                         message_type: SupportedMessageTypes,
                         task_type: str | None,
                         task_payload: dict | None):
    # The _generate_frames() method is only used when C++ mode is disabled
    endpoint = '/test'
    port = 8088
    method = HTTPMethod.POST
    accept_status = HTTPStatus.OK
    url = make_url(port, endpoint)

    df: pd.DataFrame = dataset_pandas['filter_probs.csv']

    if lines:
        content_type = MimeTypes.TEXT.value
    else:
        content_type = MimeTypes.JSON.value

    if use_payload_to_df_fn:
        mock_results = df[['v2', 'v3']].copy(deep=True)
        payload_to_df_fn = mock.MagicMock(return_value=mock_results)
    else:
        payload_to_df_fn = None

    buf = df_to_stream_json(df, StringIO(), lines=lines)
    buf.seek(0)

    payload = buf.read()

    stage = HttpServerSourceStage(config=config,
                                  port=port,
                                  endpoint=endpoint,
                                  method=method,
                                  accept_status=accept_status,
                                  lines=lines,
                                  payload_to_df_fn=payload_to_df_fn,
                                  message_type=message_type,
                                  task_type=task_type,
                                  task_payload=task_payload)

    if not use_payload_to_df_fn:
        stage._set_default_payload_to_df_fn()

    generate_frames = stage._generate_frames(mock_subscription)
    msg_queue = queue.SimpleQueue()

    get_next_thread = GetNext(msg_queue, generate_frames)
    get_next_thread.start()

    attempt = 0
    while not stage._processing and get_next_thread.is_alive() and attempt < 2:
        time.sleep(0.1)
        attempt += 1

    assert stage._processing
    assert get_next_thread.is_alive()

    # allow retries for more robust testing
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)

    response = session.request(method=method.value,
                               url=url,
                               data=payload,
                               timeout=10,
                               allow_redirects=False,
                               headers={
                                   "Content-Type": content_type, "unit": "test"
                               })

    result_msg = msg_queue.get(timeout=5.0)
    get_next_thread.join()

    assert response.status_code == accept_status.value
    assert response.headers["Content-Type"] == MimeTypes.TEXT.value
    assert response.text == ""

    if use_payload_to_df_fn:
        payload_to_df_fn.assert_called_once_with(payload, lines)
        expected_df = df[['v2', 'v3']]
    else:
        expected_df = df

    if message_type == SupportedMessageTypes.CONTROL_MESSAGE:
        expected_class = ControlMessage
        actual_df = result_msg.payload().df
    else:
        expected_class = MessageMeta
        actual_df = result_msg.df

    assert isinstance(result_msg, expected_class)
    dataset_pandas.assert_compare_df(expected_df, actual_df)

    if message_type == SupportedMessageTypes.CONTROL_MESSAGE:
        if task_type is not None:
            expected_tasks = {task_type: deque([task_payload])}
        else:
            expected_tasks = {}

        assert result_msg.get_tasks() == expected_tasks

        # Subset of headers that we want to check for
        expected_headers = {
            'Host': f'127.0.0.1:{port}',
            'endpoint': endpoint,
            'method': method.value,
            'remote_address': '127.0.0.1',
            'unit': 'test'
        }

        actual_headers = result_msg.get_metadata()['http_fields']
        for (key, value) in expected_headers.items():
            assert actual_headers[key] == value


@pytest.mark.parametrize("invalid_method", [HTTPMethod.GET, HTTPMethod.PATCH])
def test_constructor_invalid_method(config: Config, invalid_method: HTTPMethod):
    with pytest.raises(ValueError):
        HttpServerSourceStage(config=config, method=invalid_method)


@pytest.mark.parametrize("invalid_accept_status", [HTTPStatus.CONTINUE, HTTPStatus.FOUND])
def test_constructor_invalid_accept_status(config: Config, invalid_accept_status: HTTPStatus):
    with pytest.raises(ValueError):
        HttpServerSourceStage(config=config, accept_status=invalid_accept_status)


@pytest.mark.slow
@pytest.mark.cpu_mode
@pytest.mark.parametrize(
    "lines",
    [False, pytest.param(True, marks=pytest.mark.skip(reason="https://github.com/rapidsai/cudf/issues/15820"))],
    ids=["json", "lines"])
@pytest.mark.parametrize("use_payload_to_df_fn", [False, True], ids=["no_payload_to_df_fn", "payload_to_df_fn"])
def test_parse_errors(config: Config, mock_subscription: mock.MagicMock, lines: bool, use_payload_to_df_fn: bool):
    expected_status = HTTPStatus.BAD_REQUEST

    endpoint = '/test'
    port = 8088
    method = HTTPMethod.POST
    accept_status = HTTPStatus.OK
    url = make_url(port, endpoint)

    if lines:
        content_type = MimeTypes.TEXT.value
    else:
        content_type = MimeTypes.JSON.value

    if use_payload_to_df_fn:
        payload_to_df_fn = mock.MagicMock(side_effect=ValueError("Invalid payload"))
    else:
        payload_to_df_fn = None

    payload = '{"not_valid":"json'

    stage = HttpServerSourceStage(config=config,
                                  port=port,
                                  endpoint=endpoint,
                                  method=method,
                                  accept_status=accept_status,
                                  lines=lines,
                                  payload_to_df_fn=payload_to_df_fn)

    generate_frames = stage._generate_frames(mock_subscription)
    msg_queue = queue.SimpleQueue()

    get_next_thread = GetNext(msg_queue, generate_frames)
    get_next_thread.start()

    attempt = 0
    while not stage._processing and get_next_thread.is_alive() and attempt < 2:
        time.sleep(0.1)
        attempt += 1

    assert stage._processing
    assert get_next_thread.is_alive()

    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)

    response = session.request(method=method.value,
                               url=url,
                               data=payload,
                               timeout=10.0,
                               allow_redirects=False,
                               headers={"Content-Type": content_type})

    assert msg_queue.empty()
    assert get_next_thread.is_alive()

    assert response.status_code == expected_status.value
    assert response.headers["Content-Type"] == MimeTypes.TEXT.value
    assert "error" in response.text.lower()  # just verify that we got some sort of error message

    if use_payload_to_df_fn:
        payload_to_df_fn.assert_called_once_with(payload, lines)

    # get_next_thread will block until it processes a valid message or the queue is closed
    stage._queue.close()

    with pytest.raises(StopIteration):
        get_next_thread.join()
