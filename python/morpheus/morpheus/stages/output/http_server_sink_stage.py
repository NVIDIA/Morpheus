# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
"""Sink stage that starts an HTTP server and listens for incoming requests on a specified endpoint."""

import logging
import os
import queue
import typing
from functools import partial
from http import HTTPStatus
from io import StringIO

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.io import serializers
from morpheus.messages import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.http_utils import HTTPMethod
from morpheus.utils.http_utils import HttpParseResponse
from morpheus.utils.http_utils import MimeTypes
from morpheus.utils.type_aliases import DataFrameType

logger = logging.getLogger(__name__)


@register_stage("to-http-server", ignore_args=["df_serializer_fn"])
class HttpServerSinkStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Sink stage that starts an HTTP server and listens for incoming requests on a specified endpoint.

    Incoming messages are queued up to `max_queue_size`, when an HTTP client makes a request messages are read from the
    queue up to `max_rows_per_response` and returned.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration instance.
    bind_address : str, default "127.0.0.1"
        The address to bind the HTTP server to.
    port : int, default 8080
        The port to bind the HTTP server to.
    endpoint : str, default "/"
        The endpoint to listen for requests on.
    method : `morpheus.utils.http_utils.HTTPMethod`, optional, case_sensitive = False
        HTTP method to listen for.
    max_queue_size : int, default None
        Maximum number of requests to queue before rejecting requests. If `None` then `config.edge_buffer_size` will be
        used. Once the queue is full, the incoming edge buffer will begin to fill up.
    num_server_threads : int, default None
        Number of threads to use for the HTTP server. If `None` then `len(os.sched_getaffinity(0))` will be used.
    max_rows_per_response : int, optional
        Maximum number of rows to include in a single response, by default 10000.
    overflow_pct: float, optional
        The stage stores incoming dataframes in a queue. If the received dataframes are smaller than
        `max_rows_per_response * overflow_pct`, then additional dataframes are popped off of the queue.
        Setting a higher number (0.9 or 1) can potentially improve performance by allowing as many dataframes to be
        concatinated as possible into a single response, but with the possibility of returning a response containing
        more than `max_rows_per_response` rows. Setting a lower number (0.5 or 0.75) decreases the chance, and a value
        of `0` prevents this possibility entirely.
    request_timeout_secs : int, default 30
        The maximum amount of time in seconds for any given request.
    lines : bool, default False
        If False, dataframes will be serialized to a JSON array of objects. If True, the dataframes will be
        serialized to a string JSON objects separated by end-of-line characters.
        Ignored if `df_serializer` is provided.
    df_serializer_fn : typing.Callable[[DataFrameType], str], optional
        Optional custom dataframe serializer function.
    """

    def __init__(self,
                 config: Config,
                 bind_address: str = "127.0.0.1",
                 port: int = 8080,
                 endpoint: str = "/message",
                 method: HTTPMethod = HTTPMethod.GET,
                 max_queue_size: int = None,
                 num_server_threads: int = None,
                 max_rows_per_response: int = 10000,
                 overflow_pct: float = 0.75,
                 request_timeout_secs: int = 30,
                 lines: bool = False,
                 df_serializer_fn: typing.Callable[[DataFrameType], str] = None):
        super().__init__(config)
        self._bind_address = bind_address
        self._port = port
        self._endpoint = endpoint
        self._method = method
        self._num_server_threads = num_server_threads or len(os.sched_getaffinity(0))
        self._max_rows_per_response = max_rows_per_response
        self._overflow_pct = overflow_pct
        self._request_timeout_secs = request_timeout_secs
        self._lines = lines

        if self._lines:
            self._content_type = MimeTypes.TEXT.value
        else:
            self._content_type = MimeTypes.JSON.value

        self._df_serializer_fn = df_serializer_fn or self._default_df_serializer

        self._df_pkg = self.get_df_pkg()

        # FiberQueue doesn't have a way to check the size, nor does it have a way to check if it's empty without
        # attempting to perform a read. We'll keep track of the size ourselves.
        self._queue = queue.Queue(maxsize=max_queue_size or config.edge_buffer_size)
        self._server = None

    @property
    def name(self) -> str:
        """Unique stage name."""
        return "to-http-server"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple(`morpheus.pipeline.messages.MessageMeta`, )
            Accepted input types.

        """
        return (MessageMeta, )

    def supports_cpp_node(self):
        """Indicates whether or not this stage supports a C++ node."""
        return False

    async def start_async(self):
        """Starts the HTTP server."""
        from morpheus.common import HttpEndpoint
        from morpheus.common import HttpServer
        http_endpoint = HttpEndpoint(py_parse_fn=self._request_handler, url=self._endpoint, method=self._method.name)
        self._server = HttpServer(endpoints=[http_endpoint],
                                  bind_address=self._bind_address,
                                  port=self._port,
                                  num_threads=self._num_server_threads,
                                  request_timeout=self._request_timeout_secs)
        self._server.start()

    def is_running(self) -> bool:
        """Indicates whether or not the stage is running."""
        if self._server is not None:
            return self._server.is_running()

        return False

    def _default_df_serializer(self, df: DataFrameType) -> str:
        """
        Default dataframe serializer, used when `df_serializer_fn` is not provided.
        """
        str_buf = StringIO()
        serializers.df_to_stream_json(df=df, stream=str_buf, lines=self._lines)
        return str_buf.getvalue()

    def _request_callback(self, df: DataFrameType, num_tasks: int, has_error: bool, error_msg: str) -> None:
        try:
            if has_error:
                logger.error(error_msg)

                # If the client failed to read the response, then we need to put the dataframe back into the queue
                self._queue.put(df)

            # Even in the event of an error, we need to mark the tasks as done.
            for _ in range(num_tasks):
                self._queue.task_done()
        except Exception as e:
            logger.error("Unknown error in request callback: %s", e)

    def _request_handler(self, _: str) -> HttpParseResponse:
        num_rows = 0
        data_frames = []
        try:
            while (num_rows == 0 or num_rows < (self._max_rows_per_response * self._overflow_pct)):
                df = self._queue.get_nowait()
                num_rows += len(df)
                data_frames.append(df)
        except queue.Empty:
            pass
        except Exception as e:
            err_msg = "Unknown error processing request"
            logger.error(f"{err_msg}: %s", e)
            return HttpParseResponse(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                                     content_type=MimeTypes.TEXT.value,
                                     body=err_msg)

        if (len(data_frames) > 0):
            if len(data_frames) > 1:
                df = self._df_pkg.concat(data_frames)
            else:
                df = data_frames[0]

            return HttpParseResponse(status_code=HTTPStatus.OK.value,
                                     content_type=self._content_type,
                                     body=self._df_serializer_fn(df),
                                     on_complete_callback=partial(self._request_callback, df, len(data_frames)))

        return HttpParseResponse(status_code=HTTPStatus.NO_CONTENT.value, content_type=MimeTypes.TEXT.value, body="")

    def _partition_df(self, df: DataFrameType) -> typing.Iterable[DataFrameType]:
        """
        Partition a dataframe into slices no larger than `self._max_rows_per_response`.

        Parameters
        ----------
        df : DataFrameType
            DataFrame to partition
        """
        slice_start = 0
        while (slice_start < len(df)):
            slice_end = min(slice_start + self._max_rows_per_response, len(df))
            df_slice = df.iloc[slice_start:slice_end]

            yield df_slice

            slice_start = slice_end

    def _process_message(self, msg: MessageMeta) -> MessageMeta:
        # In order to conform to the `self._max_rows_per_response` argument we need to slice up the dataframe here
        # because our queue isn't a deque.
        for df_slice in self._partition_df(msg.df):
            # We want to block, such that if the queue is full, we want our edgebuffer to start filling up.
            self._queue.put(df_slice, block=True)

        return msg

    def _block_until_empty(self):
        logger.debug("Waiting for queue to empty")
        self._queue.join()
        logger.debug("stopping server")
        self._server.stop()
        logger.debug("stopped")
        self._server = None

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name,
                                 ops.map(self._process_message),
                                 ops.on_completed(self._block_until_empty))
        builder.make_edge(input_node, node)

        return node
