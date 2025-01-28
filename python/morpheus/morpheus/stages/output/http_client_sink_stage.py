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
"""Write all messages to an HTTP endpoint."""

import logging
import typing
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
from morpheus.utils import http_utils
from morpheus.utils.http_utils import HTTPMethod
from morpheus.utils.http_utils import MimeTypes
from morpheus.utils.type_aliases import DataFrameType

logger = logging.getLogger(__name__)


@register_stage("to-http", ignore_args=["query_params", "headers", "df_to_request_kwargs_fn", "**request_kwargs"])
class HttpClientSinkStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Write all messages to an HTTP endpoint.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    base_url : str
        Server base url, should include the intended protocol (e.g. http:// or https://) and port if necessary.
        This may or may not include a base path from which `endpoint` will be appended.
        examples:
        * "https://nvcr.io/"
        * "http://localhost:8080/base/path"
    endpoint : str
        Endpoint to which messages will be sent. This will be appended to `base_url` and may include a query string.
        The primary difference between `endpoint` and `base_url` is that `endpoint` may contain named format strings,
        when `static_endpoint` is `False`, and thus could potentially be different for each request.

        Format strings which will be replaced with the corresponding column value from the first row of the incoming
        dataframe, if no such column exists a `ValueError` will be raised. When `endpoint` contains query a query string
        this has the potential of allowing for the values of the query string to be different for each request. When
        `query_params` is not `None` the values in `query_params` will be appended to the query string. This could
        potentially result in duplicate keys in the query string, some servers support this transforming duplicate keys
        into an array of values (ex "?t=1&t=2" => "t=[1,2]"), others do not.

        Note: When `max_rows_per_payload=1`, this has the effect of producing a separate request for each row in the
        dataframe potentially using a unique endpoint for each request.

        If additional customizations are required, `df_to_request_kwargs_fn` can be used to perform additional
        customizations of the request.

        examples:
        * "api/v1/endpoint"
        * "api/v1/endpoint?time={timestamp}&id={id}"
        * "/{model_name}/{user}?time={timestamp}"
    static_endpoint : bool, default True
        Setting this to `True` indicates that the value of `endpoint` does not change between requests, and can be an
        optimization.
    headers : dict, optional
        Optional set of headers to include in the request.
        If `None` the header value will be inferred based on `lines`.
        * `{"Content-Type": "text/plain"}` when `lines` is `True`
        * `{"Content-Type": "application/json"}` when `lines` is `False`
    query_params : dict, optional
        Optional set of query parameters to include in the request.
    method : `morpheus.utils.http_utils.HTTPMethod`, optional, case_sensitive = False
        HTTP method to use when sending messages, by default "POST". Currently only "POST", "PUT" and "PATCH" are
        supported.
    error_sleep_time : float, optional
        Amount of time in seconds to sleep after the client receives an error.
        The client will perform an exponential backoff starting at `error_sleep_time`.
        Setting this to 0 causes the client to retry the request as fast as possible.
        If the server sets a `Retry-After` header and `respect_retry_after_header` is `True`, then that value will take
        precedence over `error_sleep_time`.
    respect_retry_after_header : bool, optional
        If True, the client will respect the `Retry-After` header if it is set by the server. If False, the client will
        perform an exponential backoff starting at `error_sleep_time`.
    request_timeout_secs : int, optional
        Number of seconds to wait for the server to send data before giving up and raising an exception.
    accept_status_codes :  typing.Iterable[int], optional,  multiple = True
        List of acceptable status codes, by default (200, 201, 202).
    max_retries : int, default 10
        Maximum number of times to retry the request fails, receives a redirect or returns a status in the
        `retry_status_codes` list. Setting this to 0 disables this feature, and setting this to a negative number will
        raise a `ValueError`.
    max_rows_per_payload : int, optional
        Maximum number of rows to include in a single payload, by default 10000.
        Setting this to 1 will send each row as a separate request.
    lines : bool, default False
        If False, dataframes will be serialized to a JSON array of objects. If True, then the dataframes will be
        serialized to a string JSON objects separated by end-of-line characters.
    df_to_request_kwargs_fn: typing.Callable[[str, str, DataFrameType], dict], optional
        Optional function to perform additional customizations of the request. This function will be called for each
        DataFrame (according to `max_rows_per_payload`) before the request is sent.
        The function will be called with the following arguments:
        * `base_url` : str
        * `endpoint` : str
        * `df` : DataFrameType

        The function should return a dict containing any keyword argument expected by the `requests.Session.request`
        function:
        https://requests.readthedocs.io/en/v2.9.1/api/#requests.Session.request

        Specifically, this function is responsible for serializing the DataFrame to either a POST/PUT body or a query
        string. This method has the potential of returning a value for `url` overriding the value of `endpoint` and
        `base_url`, even when `static_endpoint` is True.
    **request_kwargs : dict
        Additional arguments to pass to the `requests.Session.request` function. These values will are potentially
        overridden by the results of `df_to_request_kwargs_fn` if it is not `None`, otherwise the value of `data` will
        be overwritten, as will `url` when `static_endpoint` is False.
    """

    def __init__(self,
                 c: Config,
                 base_url: str,
                 endpoint: str,
                 static_endpoint: bool = True,
                 headers: dict = None,
                 query_params: dict = None,
                 method: HTTPMethod = HTTPMethod.POST,
                 error_sleep_time: float = 0.1,
                 respect_retry_after_header: bool = True,
                 request_timeout_secs: int = 30,
                 accept_status_codes: typing.Iterable[int] = (
                     HTTPStatus.OK,
                     HTTPStatus.CREATED,
                     HTTPStatus.ACCEPTED,
                 ),
                 max_retries: int = 10,
                 max_rows_per_payload: int = 10000,
                 lines: bool = False,
                 df_to_request_kwargs_fn: typing.Optional[typing.Callable[[str, str, DataFrameType], dict]] = None,
                 **request_kwargs):
        super().__init__(c)
        self._base_url = http_utils.prepare_url(base_url)

        if (callable(endpoint) and static_endpoint):
            raise ValueError("endpoint must be a string when static_endpoint is True")

        self._endpoint = endpoint
        self._query_params = query_params

        if headers is None:
            if lines:
                headers = {"Content-Type": MimeTypes.TEXT.value}
            else:
                headers = {"Content-Type": MimeTypes.JSON.value}

        self._headers = headers

        self._method = method

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
        self._static_endpoint = static_endpoint
        self._max_rows_per_payload = max_rows_per_payload
        self._lines = lines
        self._df_to_request_kwargs_fn = df_to_request_kwargs_fn
        self._requst_kwargs = request_kwargs
        self._http_session = None

    @property
    def name(self) -> str:
        """Unique name of the stage."""
        return "to-http"

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
        """Indicates whether this stage supports CPP nodes."""
        return False

    def _df_to_url(self, df: DataFrameType) -> str:
        """
        Convert a Dataframe to a URL. Only called when self._static_endpoint is False.

        Parameters
        ----------
        df : DataFrameType
            DataFrame to infer enpoint from.

        Returns
        -------
        str
            URL.
        """
        endpoint = self._endpoint.format(**df.iloc[0].to_dict())
        return f"{self._base_url}{endpoint}"

    def _df_to_payload(self, df: DataFrameType) -> StringIO:
        str_buf = StringIO()
        serializers.df_to_stream_json(df=df, stream=str_buf, lines=self._lines)
        str_buf.seek(0)
        return str_buf

    def _chunk_requests(self, df: DataFrameType) -> typing.Iterable[dict]:
        """
        Convert a Dataframe to a series of urls and payloads with no more than `self._max_rows_per_payload`.

        Parameters
        ----------
        df : DataFrameType
            DataFrame to chunk, and convert to urls and payloads.
        """
        slice_start = 0
        while (slice_start < len(df)):
            slice_end = min(slice_start + self._max_rows_per_payload, len(df))
            df_slice = df.iloc[slice_start:slice_end]

            if self._df_to_request_kwargs_fn is not None:
                yield self._df_to_request_kwargs_fn(self._base_url, self._endpoint, df_slice)
            else:
                chunk = {'data': self._df_to_payload(df_slice)}
                if not self._static_endpoint:
                    chunk['url'] = self._df_to_url(df_slice)

                yield chunk

            slice_start = slice_end

    def _process_message(self, msg: MessageMeta) -> MessageMeta:

        request_args = {
            'method': self._method.value,
            'headers': self._headers,
            'timeout': self._request_timeout_secs,
            'params': self._query_params
        }

        if self._static_endpoint:
            request_args['url'] = f"{self._base_url}{self._endpoint}"

        request_args.update(self._requst_kwargs)

        for chunk in self._chunk_requests(msg.df):
            request_args.update(chunk)
            http_utils.request_with_retry(request_args,
                                          requests_session=self._http_session,
                                          max_retries=self._max_retries,
                                          sleep_time=self._error_sleep_time,
                                          respect_retry_after_header=self._respect_retry_after_header,
                                          accept_status_codes=self._accept_status_codes)

        return msg

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self._process_message))
        builder.make_edge(input_node, node)

        return node
