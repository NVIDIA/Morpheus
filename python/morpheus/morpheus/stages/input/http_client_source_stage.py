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
"""Source stage that polls a remote HTTP server for incoming data."""

import logging
import time
import typing
from http import HTTPStatus

import mrc
import requests

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.io.utils import get_json_reader
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.configurable_output_source import ConfigurableOutputSource
from morpheus.pipeline.configurable_output_source import SupportedMessageTypes
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.utils import http_utils
from morpheus.utils.type_aliases import DataFrameType

logger = logging.getLogger(__name__)


@register_stage("from-http-client", ignore_args=["query_params", "headers", "**request_kwargs"])
class HttpClientSourceStage(GpuAndCpuMixin, PreallocatorMixin, ConfigurableOutputSource):
    """
    Source stage that polls a remote HTTP server for incoming data.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration instance.
    url : str
        Remote URL to poll for data, ex `https://catalog.ngc.nvidia.com/api/collections`. This should include protocol
        prefix (ex. "http://", "https://") and port if necessary. If the protocol is omitted, `http://` will be used.
    query_params : dict, callable, default None
        Query parameters to pass to the remote URL. Can either be a dictionary of key-value pairs or a callable that
        returns a dictionary of key-value pairs. If a callable is provided, it will be called with no arguments.
    headers: dict, default None
        Headers sent with the request.
    method : `morpheus.utils.http_utils.HTTPMethod`, optional, case_sensitive = False
        HTTP method to use.
    sleep_time : float, default 0.1
        Amount of time in seconds to sleep between successive requests. Setting this to 0 disables this feature.
    error_sleep_time : float, default 0.1
        Amount of time in seconds to sleep after the client receives an error.
        The client will perform an exponential backoff starting at `error_sleep_time`.
        Setting this to 0 causes the client to retry the request as fast as possible.
        If the server sets a `Retry-After` header and `respect_retry_after_header` is `True`, then that value will take
        precedence over `error_sleep_time`.
    respect_retry_after_header: bool, default True
        If True, the client will respect the `Retry-After` header if it is set by the server. If False, the client will
        perform an exponential backoff starting at `error_sleep_time`.
    request_timeout_secs : int, optional
        Number of seconds to wait for the server to send data before giving up and raising an exception.
    max_errors : int, default 10
        Maximum number of consequtive errors to receive before raising an error.
    accept_status_codes : typing.Iterable[int], optional,  multiple = True
        List of status codes to accept. If the response status code is not in this collection, then the request will be
        considered an error
    max_retries : int, default 10
        Maximum number of times to retry the request fails, receives a redirect or returns a status in the
        `retry_status_codes` list. Setting this to 0 disables this feature, and setting this to a negative number will
        raise a `ValueError`.
    lines : bool, default False
        If False, the response payloads are expected to be a JSON array of objects. If True, the payloads are expected
        to contain a JSON objects separated by end-of-line characters.
    stop_after : int, default 0
        Stops ingesting after emitting `stop_after` records (rows in the dataframe). Useful for testing. Disabled if `0`
    payload_to_df_fn : callable, default None
        A callable that takes the HTTP payload bytes as the first argument and the `lines` parameter is passed in as
        the second argument and returns a DataFrame. If unset `cudf.read_json` is used in GPU mode and
        `pandas.read_json` in CPU mode.
    message_type : `SupportedMessageTypes`, case_sensitive = False
        The type of message to emit.
    task_type : str, default = None
        If specified, adds the specified task to the `ControlMessage`. This parameter is only valid when `message_type`
        is set to `CONTROL_MESSAGE`. If not `None`, `task_payload` must also be specified.
    task_payload : dict, default = None
        If specified, adds the specified task to the `ControlMessage`. This parameter is only valid when `message_type`
        is set to `CONTROL_MESSAGE`. If not `None`, `task_type` must also be specified.
    **request_kwargs : dict
        Additional arguments to pass to the `requests.request` function.
    """

    def __init__(self,
                 config: Config,
                 url: str,
                 query_params: typing.Union[dict, typing.Callable] = None,
                 headers: dict = None,
                 method: http_utils.HTTPMethod = http_utils.HTTPMethod.GET,
                 sleep_time: float = 0.1,
                 error_sleep_time: float = 0.1,
                 respect_retry_after_header: bool = True,
                 request_timeout_secs: int = 30,
                 accept_status_codes: typing.Iterable[int] = (HTTPStatus.OK, ),
                 max_retries: int = 10,
                 lines: bool = False,
                 stop_after: int = 0,
                 payload_to_df_fn: typing.Callable[[bytes, bool], DataFrameType] = None,
                 message_type: SupportedMessageTypes = SupportedMessageTypes.MESSAGE_META,
                 task_type: str = None,
                 task_payload: dict = None,
                 **request_kwargs):
        super().__init__(config, message_type=message_type, task_type=task_type, task_payload=task_payload)
        self._url = http_utils.prepare_url(url)

        if callable(query_params):
            self._query_params_fn = query_params
            self._query_params = None
        else:
            self._query_params_fn = None
            self._query_params = query_params

        self._headers = headers

        self._method = method

        if sleep_time >= 0:
            self._sleep_time = sleep_time
        else:
            raise ValueError("sleep_time must be >= 0")

        if error_sleep_time >= 0:
            self._error_sleep_time = error_sleep_time
        else:
            raise ValueError("error_sleep_time must be >= 0")

        self._respect_retry_after_header = respect_retry_after_header

        self._request_timeout_secs = request_timeout_secs
        if max_retries >= 0:
            self._max_retries = max_retries
        else:
            raise ValueError("max_retries must be >= 0")

        self._accept_status_codes = tuple(accept_status_codes)

        self._stop_after = stop_after
        self._lines = lines
        self._requst_kwargs = request_kwargs

        if payload_to_df_fn is not None:
            self._payload_to_df_fn = payload_to_df_fn
        else:
            reader = get_json_reader(self._config.execution_mode)
            self._payload_to_df_fn = lambda payload, lines: reader(payload, lines=lines)

    @property
    def name(self) -> str:
        """Unique name of the stage"""
        return "from-http-client"

    def supports_cpp_node(self) -> bool:
        """Indicates whether or not this stage supports a C++ implementation"""
        return False

    def _parse_response(self, response: requests.Response) -> typing.Union[DataFrameType, None]:
        """
        Returns a DataFrame parsed from the response payload. If the response payload is empty, then `None` is returned.
        """
        payload = response.content

        return self._payload_to_df_fn(payload, self._lines)

    def _generate_frames(self, subscription: mrc.Subscription) -> typing.Iterator[MessageMeta]:
        # Running counter of the number of messages emitted by this source
        num_records_emitted = 0

        # The http_session variable is an in/out argument for the requests_retry_wrapper.request function and will be
        # initialized on the first call
        http_session = None

        request_args = {
            'method': self._method.value,
            'url': self._url,
            'headers': self._headers,
            'timeout': self._request_timeout_secs,

            # when self._query_params_fn this will be overwritten on each iteration
            'params': self._query_params
        }

        request_args.update(self._requst_kwargs)

        while (not self.is_stop_requested() and subscription.is_subscribed()
               and (self._stop_after == 0 or num_records_emitted < self._stop_after)):
            if self._query_params_fn is not None:
                request_args['params'] = self._query_params_fn()

            (http_session,
             response) = http_utils.request_with_retry(request_args,
                                                       requests_session=http_session,
                                                       max_retries=self._max_retries,
                                                       sleep_time=self._error_sleep_time,
                                                       respect_retry_after_header=self._respect_retry_after_header,
                                                       accept_status_codes=self._accept_status_codes)

            df = self._parse_response(response)
            # Even if we didn't receive any errors, the server may not have had any data for us.
            if df is not None and len(df):
                num_rows = len(df)
                msg_meta = MessageMeta(df)
                if self._message_type is SupportedMessageTypes.CONTROL_MESSAGE:
                    http_fields = request_args.copy()
                    http_fields.update(response.headers)

                    out_msg = ControlMessage({"metadata": {"http_fields": http_fields}})
                    out_msg.payload(msg_meta)
                    if self._task_type is not None:
                        out_msg.add_task(self._task_type, self._task_payload)
                else:
                    out_msg = msg_meta

                yield out_msg
                num_records_emitted += num_rows

            time.sleep(self._sleep_time)

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:
        return builder.make_source(self.unique_name, self._generate_frames)
