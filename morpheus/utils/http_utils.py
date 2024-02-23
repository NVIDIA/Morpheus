# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
"""HTTP utilities"""

import logging
import time
import typing
from enum import Enum
from http import HTTPStatus

import requests
import urllib3

logger = logging.getLogger(__name__)


class MimeTypes(Enum):
    """Not a complete list of mime types, just the ones we use."""
    TEXT = "text/plain"
    JSON = "application/json"


class HTTPMethod(Enum):
    """Not a complete list of HTTP methods, just the ones we use."""
    GET = "GET"
    PATCH = "PATCH"
    POST = "POST"
    PUT = "PUT"


HttpOnCompleteCallbackFn = typing.Callable[[bool, str], None]
"""
Optional callback function invoked by `morpheus.common.HttpServer` once a response is completed,
either successfully or encountered a failure.

Parameters
----------
has_error: bool
    `False` if the response was successfully sent to the client, `True` if an error was encountered.

error_message: str
    When `has_error` is `True`, this will contain the error message. Otherwise, it will be an empty string.
"""


class HttpParseResponse(typing.NamedTuple):
    """
    A tuple consisting of the HTTP status code, mime type to be used for the Content-Type header, the body of the
    response and an optional callback function to be invoked once the response is completed.

    The values for `status_code` and `content_type` are strings rather than `http.HTTPStatus` and `MimeTypes` because
    these are intended to be consumed directly by the C++ implementation of `morpheus.common.HttpServer`. Instead these
    enums should be used to construct `HttpParseResponse` instances:

    >>> http_parse_response = HttpParseResponse(status_code=HTTPStatus.OK.value,
    ...                                         content_type=MimeTypes.TEXT.value,
    ...                                         body="OK",
    ...                                         on_complete_callback=None)
    >>>
    """
    status_code: int
    content_type: str
    body: str
    on_complete_callback: typing.Optional[HttpOnCompleteCallbackFn] = None


# pylint: disable=inconsistent-return-statements
def request_with_retry(
    request_kwargs: dict,
    requests_session: typing.Optional[requests.Session] = None,
    max_retries: int = 10,
    sleep_time: float = 0.1,
    accept_status_codes: typing.Iterable[HTTPStatus] = (HTTPStatus.OK, ),
    respect_retry_after_header: bool = True,
    on_success_fn: typing.Optional[typing.Callable] = None
) -> typing.Tuple[requests.Session, typing.Union[requests.Response, typing.Any]]:
    """
    Wrapper around `requests.request` that retries on failure.

    This code is a work-around for an issue (https://github.com/urllib3/urllib3/issues/2751), in urllib3's Retry class
    and should be removed once it is resolved.

    Upon successfull completion, the `on_success_fn` is called (if not `None`) with the response object.
    When `on_success_fn` is `None`, a tuple containing the request session and the response object is returned,
    otherwise a tuple containing the request session and the return value of `on_success_fn` is returned.

    If `on_success_fn` raises an exception, it is treated as a failure and the request is retried.
    """

    try_count = 0
    while try_count <= max_retries:
        if requests_session is None:
            requests_session = requests.Session()

        # set to an int if the response has a Retry-After header and `respect_retry_after_header` is True
        retry_after_header = None

        try:
            response = requests_session.request(**request_kwargs)
            if response.status_code in accept_status_codes:
                if on_success_fn is not None:
                    return (requests_session, on_success_fn(response))

                return (requests_session, response)

            if respect_retry_after_header and 'Retry-After' in response.headers:
                retry_after_header = int(response.headers['Retry-After'])

            raise RuntimeError(f"Received unexpected status code {response.status_code}: {response.text}")
        except Exception as e:
            # if we got a requests exception, close the session, so that on a retry we get a new connection
            if isinstance(e, requests.exceptions.RequestException):
                try:
                    requests_session.close()
                finally:
                    requests_session = None

            try_count += 1

            if try_count >= max_retries:
                logger.error("Failed after %s retries: %s", max_retries, e)
                raise e

            if retry_after_header is not None:
                actual_sleep_time = retry_after_header
            else:
                actual_sleep_time = (2**(try_count - 1)) * sleep_time

            logger.error("Error occurred performing %s request to %s: %s",
                         request_kwargs['method'],
                         request_kwargs['url'],
                         e)
            logger.debug("Sleeping for %s seconds before retrying request again", actual_sleep_time)
            time.sleep(actual_sleep_time)


def prepare_url(url: str) -> str:
    """
    Verifies that `url` contains a protocol scheme and a host and returns the url.
    If no protocol scheme is provided, `http` is used.
    """
    parsed_url = urllib3.util.parse_url(url)
    if parsed_url.scheme is None or parsed_url.host is None:
        url = f"http://{url}"

        parsed_url = urllib3.util.parse_url(url)
        if parsed_url.scheme is None or parsed_url.host is None:
            raise ValueError(f"Invalid URL: {url}")

        logger.warning("No protocol scheme provided in URL, using: %s", url)

    return parsed_url.url


_T = typing.TypeVar('_T')
_P = typing.ParamSpec('_P')


def retry_async(retry_exceptions=None):
    import tenacity

    def inner(func: typing.Callable[_P, _T]) -> typing.Callable[_P, _T]:

        @tenacity.retry(wait=tenacity.wait_exponential_jitter(0.1),
                        stop=tenacity.stop_after_attempt(10),
                        retry=tenacity.retry_if_exception_type(retry_exceptions),
                        reraise=True)
        async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            return await func(*args, **kwargs)

        return wrapper

    return inner
