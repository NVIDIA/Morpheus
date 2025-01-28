# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
from collections.abc import Callable
from functools import wraps

from morpheus.messages import ControlMessage

T = typing.TypeVar('T')
CM_SKIP_P = typing.ParamSpec('CM_SKIP_P')


def cm_skip_processing_if_failed(func: Callable[CM_SKIP_P, T]) -> Callable[CM_SKIP_P, T]:
    """
    Decorator function to skip processing if the ControlMessage has failed.

    Parameters
    ----------
    func : typing.Callable
        The function to decorate.

    Returns
    -------
    wrapper : typing.Callable
        The decorated function.
    """

    @wraps(func)
    def wrapper(control_message: ControlMessage, *args: CM_SKIP_P.args, **kwargs: CM_SKIP_P.kwargs) -> T:
        if (control_message.has_metadata("cm_failed") and control_message.get_metadata("cm_failed")):
            return control_message

        return func(control_message, *args, **kwargs)

    return wrapper


def cm_ensure_payload_not_null(control_message: ControlMessage):
    """
    Ensures that the payload of a ControlMessage is not None.

    Parameters
    ----------
    control_message : ControlMessage
        The ControlMessage to check.

    Raises
    ------
    ValueError
        If the payload is None.
    """

    if (control_message.payload().mutable_dataframe() is None):
        raise ValueError("Payload cannot be None")


def cm_default_failure_context_manager(raise_on_failure: bool = False) -> typing.Callable:
    """
    Decorator function to set the default failure context manager for ControlMessage processing.

    Parameters
    ----------
    raise_on_failure : bool, optional
        Whether to raise an exception on failure, by default False.

    Returns
    -------
    decorator : typing.Callable
        The decorated function.
    """

    def decorator(func):

        @wraps(func)
        def wrapper(control_message: ControlMessage, *args, **kwargs):
            ret_cm = control_message
            with CMDefaultFailureContextManager(control_message=control_message,
                                                raise_on_failure=raise_on_failure) as ctx_mgr:
                cm_ensure_payload_not_null(control_message=control_message)
                ret_cm = func(ctx_mgr.control_message, *args, **kwargs)

            return ret_cm

        return wrapper

    return decorator


def cm_set_failure(control_message: ControlMessage, reason: str) -> ControlMessage:
    """
    Sets the failure metadata on a ControlMessage.

    Parameters
    ----------
    control_message : ControlMessage
        The ControlMessage to set the failure metadata on.
    reason : str
        The reason for the failure.

    Returns
    -------
    control_message : ControlMessage
        The modified ControlMessage with the failure metadata set.
    """

    control_message.set_metadata("cm_failed", True)
    control_message.set_metadata("cm_failed_reason", reason)

    return control_message


class CMDefaultFailureContextManager:
    """
    Default Context manager for handling ControlMessage failures.

    Parameters
    ----------
    control_message : ControlMessage
        The ControlMessage to handle.
    raise_on_failure : bool, optional
        Whether to raise an exception on failure, by default False.
    """

    def __init__(self, control_message: ControlMessage, raise_on_failure: bool = False):
        self.control_message = control_message
        self.raise_on_failure = raise_on_failure

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if ((exc_type is not None) and (not self.raise_on_failure)):  # An exception occurred
            if (self.control_message is not None):
                cm_set_failure(self.control_message, str(exc_value))

            return True  # Indicate that we handled the exception

        return False  # Indicate that we did not handle the exception
