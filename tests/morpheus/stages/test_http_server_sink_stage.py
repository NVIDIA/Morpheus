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

import asyncio
import queue
import typing
from http import HTTPStatus
from io import StringIO

import pytest

from _utils import make_url
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.io.serializers import df_to_stream_json
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.http_server_sink_stage import HttpServerSinkStage
from morpheus.utils.http_utils import HTTPMethod
from morpheus.utils.http_utils import MimeTypes
from morpheus.utils.http_utils import request_with_retry
from morpheus.utils.type_aliases import DataFrameType


async def make_requests(sink: HttpServerSinkStage,
                        response_queue: queue.Queue,
                        method: HTTPMethod,
                        url: str,
                        num_requests,
                        content_type: str):
    attempt = 0
    while not sink.is_running() and attempt < 2:
        await asyncio.sleep(1)
        attempt += 1

    if not sink.is_running():
        raise RuntimeError("HttpServerSinkStage did not start")

    # Not strictly needed, but we don't have a good way of knowing when the server is ready to accept requests
    # Adding this sleep here just lowers the likely-hood of seeing a logged warning on the first failed request.
    await asyncio.sleep(0.1)

    for _ in range(num_requests):
        (_, response) = request_with_retry(
            {
                'method': method.value,
                'url': url,
                'timeout': 5.0,
                'allow_redirects': False,
                'headers': {
                    "Content-Type": content_type
                }
            },
            accept_status_codes=[HTTPStatus.OK.value],
            respect_retry_after_header=False)
        response_queue.put_nowait(response)


async def run_pipe_and_request(pipe: LinearPipeline,
                               sink: HttpServerSinkStage,
                               response_queue: queue.Queue,
                               method: HTTPMethod,
                               url: str,
                               num_requests: int,
                               content_type: str):
    await asyncio.gather(pipe.run_async(), make_requests(sink, response_queue, method, url, num_requests, content_type))


def _df_to_str(df: DataFrameType, lines: bool) -> str:
    buffer = StringIO()
    df_to_stream_json(df=df, stream=buffer, lines=lines)
    buffer.seek(0)
    return buffer.read()


def _custom_serializer(df: DataFrameType) -> str:
    return str(df.index[0])


@pytest.mark.slow
@pytest.mark.gpu_and_cpu_mode
@pytest.mark.parametrize("lines", [False, True])
@pytest.mark.parametrize("max_rows_per_response", [10000, 10])
@pytest.mark.parametrize("df_serializer_fn", [None, _custom_serializer])
def test_http_server_sink_stage_pipe(config: Config,
                                     dataset_cudf: DatasetManager,
                                     lines: bool,
                                     max_rows_per_response: int,
                                     df_serializer_fn: typing.Optional[typing.Callable]):
    port = 8088
    method = HTTPMethod.GET
    endpoint = '/test'
    url = make_url(port, endpoint)
    if lines:
        content_type = MimeTypes.TEXT.value
    else:
        content_type = MimeTypes.JSON.value

    df = dataset_cudf['filter_probs.csv']

    if max_rows_per_response > len(df):
        num_requests = 1
    else:
        num_requests = len(df) // max_rows_per_response

    expected_payloads: typing.List[typing.Tuple[str, DataFrameType]] = []
    rows_serialized = 0
    while rows_serialized < len(df):
        sliced_df = df[rows_serialized:rows_serialized + max_rows_per_response]
        if df_serializer_fn is None:
            expected_payload = _df_to_str(df=sliced_df, lines=lines)
        else:
            expected_payload = f"{rows_serialized}"

        expected_payloads.append((expected_payload, sliced_df))
        rows_serialized += max_rows_per_response

    assert len(expected_payloads) == num_requests

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [df]))
    sink = pipe.add_stage(
        HttpServerSinkStage(config,
                            port=port,
                            endpoint=endpoint,
                            method=method,
                            lines=lines,
                            max_rows_per_response=max_rows_per_response,
                            df_serializer_fn=df_serializer_fn))

    response_queue = queue.SimpleQueue()
    asyncio.run(run_pipe_and_request(pipe, sink, response_queue, method, url, num_requests, content_type))

    for i in range(num_requests):
        response = response_queue.get_nowait()

        assert response.status_code == HTTPStatus.OK.value
        assert response.headers["Content-Type"] == content_type
        assert response.text == expected_payloads[i][0]
