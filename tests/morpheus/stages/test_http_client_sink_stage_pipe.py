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

import typing
from functools import partial
from io import StringIO
from unittest import mock

import pytest

from _utils import make_mock_response
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.io.serializers import df_to_stream_json
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.http_client_sink_stage import HttpClientSinkStage
from morpheus.utils.http_utils import HTTPMethod
from morpheus.utils.http_utils import MimeTypes
from morpheus.utils.type_aliases import DataFrameType


def _df_to_buffer(df: DataFrameType, lines: bool) -> StringIO:
    buffer = StringIO()
    df_to_stream_json(df=df, stream=buffer, lines=lines)
    buffer.seek(0)
    return buffer


def _df_to_url(lines: bool, base_url: str, endpoint: str, df: DataFrameType) -> dict:
    return {'url': f"{base_url}{endpoint}/{df.index[0]}", 'data': _df_to_buffer(df=df, lines=lines)}


@pytest.mark.slow
@pytest.mark.use_pandas
@pytest.mark.parametrize("method", [HTTPMethod.POST, HTTPMethod.PUT])
@pytest.mark.parametrize("lines", [False, True])
@pytest.mark.parametrize("max_rows_per_payload", [10000, 5])
@pytest.mark.parametrize("use_df_to_url,static_endpoint", [(False, True), (False, False), (True, True)])
@mock.patch("requests.Session")
@mock.patch("time.sleep")
def test_write_to_http_stage_pipe(mock_sleep: mock.MagicMock,
                                  mock_request_session: mock.MagicMock,
                                  config: Config,
                                  dataset: DatasetManager,
                                  method: HTTPMethod,
                                  lines: bool,
                                  max_rows_per_payload: int,
                                  use_df_to_url: bool,
                                  static_endpoint: bool):
    make_mock_response(mock_request_session)
    if lines:
        expected_content_type = MimeTypes.TEXT.value
    else:
        expected_content_type = MimeTypes.JSON.value

    if use_df_to_url:
        df_to_request_kwargs_fn = partial(_df_to_url, lines)
    else:
        df_to_request_kwargs_fn = None

    if static_endpoint:
        endpoint = "/data"
    else:
        endpoint = "/data/{correlationId}?callerIpAddress={callerIpAddress}"

    df = dataset.get_df('azure_ad_logs.json', no_cache=True, parser_kwargs={'lines': False})

    if max_rows_per_payload > len(df):
        num_expected_requests = 1
    else:
        num_expected_requests = len(df) // max_rows_per_payload

    expected_payloads: typing.List[typing.Tuple[StringIO, DataFrameType]] = []
    rows_serialized = 0
    while rows_serialized < len(df):
        sliced_df = df[rows_serialized:rows_serialized + max_rows_per_payload]
        expected_payload = _df_to_buffer(df=sliced_df, lines=lines)
        expected_payloads.append((expected_payload, sliced_df))
        rows_serialized += max_rows_per_payload

    assert len(expected_payloads) == num_expected_requests

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [df]))
    pipe.add_stage(
        HttpClientSinkStage(config,
                            base_url="http://fake.nvidia.com",
                            endpoint=endpoint,
                            static_endpoint=static_endpoint,
                            method=method,
                            request_timeout_secs=42,
                            query_params={'unit': 'test'},
                            lines=lines,
                            max_rows_per_payload=max_rows_per_payload,
                            df_to_request_kwargs_fn=df_to_request_kwargs_fn))
    pipe.run()

    assert mock_request_session.request.call_count == num_expected_requests

    mocked_calls = mock_request_session.request.call_args_list
    for i, call in enumerate(mocked_calls):
        # The `data` argument is a StringIO buffer which prevents us from testing directly for equality
        called_buffer = call.kwargs['data']
        (expected_payload, sliced_df) = expected_payloads[i]
        assert called_buffer.read() == expected_payload.read()
        called_buffer.seek(0)

        if static_endpoint:
            expected_endpoint = endpoint
        else:
            expected_endpoint = endpoint.format(correlationId=sliced_df.iloc[0]['correlationId'],
                                                callerIpAddress=sliced_df.iloc[0]['callerIpAddress'])

        expected_url = f'http://fake.nvidia.com{expected_endpoint}'

        if use_df_to_url:
            expected_url = f"{expected_url}/{i*max_rows_per_payload}"

        assert call == mock.call(method=method.value,
                                 headers={"Content-Type": expected_content_type},
                                 timeout=42,
                                 params={'unit': 'test'},
                                 url=expected_url,
                                 data=called_buffer)

    mock_sleep.assert_not_called()
