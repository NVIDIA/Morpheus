# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import io
import logging
import logging.handlers
import os
import re
import time
from unittest.mock import patch

import pytest

from _utils import TEST_DIRS
from morpheus.utils.logger import LogLevels
from morpheus.utils.logger import configure_logging
from morpheus.utils.logger import deprecated_message_warning
from morpheus.utils.logger import deprecated_stage_warning
from morpheus.utils.logger import reset_logging
from morpheus.utils.logger import set_log_level


def _flush_logging_queue(logger: logging.Logger):
    # Per Python documentation, the `empty` method is not reliable, adding a safety sleep
    # https://docs.python.org/3.10/library/multiprocessing.html?highlight=queue#multiprocessing.Queue.empty
    time.sleep(0.1)
    for handler in logger.handlers:
        if isinstance(handler, logging.handlers.QueueHandler):
            while (not handler.queue.empty()):
                time.sleep(0.01)


@pytest.fixture(autouse=True)
def reset_logging_fixture(reset_logging):  # pylint: disable=unused-argument, redefined-outer-name
    yield


@patch('logging.handlers.QueueListener.stop')
def test_reset_logging(queue_listener_stop_mock):

    configure_logging(log_level=logging.INFO)

    morpheus_logger = logging.getLogger("morpheus")

    assert len(morpheus_logger.handlers) > 0

    reset_logging()

    assert len(morpheus_logger.handlers) == 0

    # Force garbage collection to ensure the QueueListener is stopped by the reset_logging function
    gc.collect()

    queue_listener_stop_mock.assert_called()


@patch('logging.handlers.QueueHandler.emit')
@pytest.mark.parametrize("log_level", LogLevels)
def test_configure_logging_from_level_default_handlers(queue_handler, log_level: type[LogLevels]):
    configure_logging(log_level=log_level.value)

    morpheus_logger = logging.getLogger("morpheus")

    assert morpheus_logger.level == log_level.value
    assert morpheus_logger.propagate is False

    morpheus_logger.info("test")

    if (log_level.value <= logging.INFO and log_level.value != logging.NOTSET):
        queue_handler.assert_called()
    else:
        queue_handler.assert_not_called()


def test_configure_logging_no_args():
    with pytest.raises(Exception, match="log_level must be specified"):
        configure_logging()


@patch('logging.handlers.RotatingFileHandler.emit')
@patch('morpheus.utils.logger.TqdmLoggingHandler.emit')
def test_configure_logging_from_file(console_handler, file_handler):

    log_config_file = os.path.join(TEST_DIRS.tests_data_dir, "logging.json")

    configure_logging(log_config_file=log_config_file)

    morpheus_logger = logging.getLogger("morpheus")

    assert morpheus_logger.level == logging.WARNING
    assert morpheus_logger.propagate is False

    morpheus_logger.debug("test")

    console_handler.assert_not_called()
    file_handler.assert_not_called()

    morpheus_logger.warning("test")

    console_handler.assert_called_once()
    file_handler.assert_called_once()


def test_configure_logging_multiple_times():
    configure_logging(log_level=logging.INFO)

    morpheus_logger = logging.getLogger("morpheus")

    assert morpheus_logger.level == logging.INFO

    # Call configure_logging again without resetting
    with pytest.raises(Exception, match="Logging has already been configured"):
        configure_logging(log_level=logging.DEBUG)

    assert morpheus_logger.level == logging.INFO


def test_configure_logging_from_file_filenotfound():
    with pytest.raises(FileNotFoundError):
        configure_logging(log_config_file="does_not_exist.json")


def test_configure_logging_custom_handlers():
    # Create a string stream for the handler
    string_stream_1 = io.StringIO()
    string_stream_2 = io.StringIO()

    new_handler_1 = logging.StreamHandler(string_stream_1)
    new_handler_2 = logging.StreamHandler(string_stream_2)

    configure_logging(new_handler_1, new_handler_2, log_level=logging.DEBUG)

    morpheus_logger = logging.getLogger("morpheus")

    morpheus_logger.debug("test")

    _flush_logging_queue(morpheus_logger)

    string_stream_1.seek(0)
    string_stream_2.seek(0)

    assert string_stream_1.getvalue() == "test\n"
    assert string_stream_2.getvalue() == "test\n"


@pytest.mark.parametrize("log_level", LogLevels)
def test_set_log_level(log_level: type[LogLevels]):
    configure_logging(log_level=logging.INFO)

    morpheus_logger = logging.getLogger("morpheus")

    assert morpheus_logger.level == logging.INFO

    set_log_level(log_level.value)

    assert morpheus_logger.level == log_level.value


def test_deprecated_stage_warning(caplog: pytest.LogCaptureFixture):

    class DummyStage():
        pass

    logger = logging.getLogger()
    caplog.set_level(logging.WARNING)
    deprecated_stage_warning(logger, DummyStage, "dummy_stage")
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "The 'DummyStage' stage ('dummy_stage') has been deprecated" in caplog.text


def test_deprecated_stage_warning_with_reason(caplog: pytest.LogCaptureFixture):

    class DummyStage():
        pass

    logger = logging.getLogger()
    caplog.set_level(logging.WARNING)
    deprecated_stage_warning(logger, DummyStage, "dummy_stage", reason="This is the reason.")
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "The 'DummyStage' stage ('dummy_stage') has been deprecated and will be removed in a future version. " \
           "This is the reason." in caplog.text


def test_deprecated_message_warning():

    class OldMessage():
        pass

    class NewMessage():
        pass

    with pytest.warns(DeprecationWarning) as warnings:
        deprecated_message_warning(OldMessage, NewMessage)

    pattern_with_version = (r"The '(\w+)' message has been deprecated and will be removed "
                            r"after version (\d+\.\d+) release. Please use '(\w+)' instead.")

    pattern_without_version = (r"The '(\w+)' message has been deprecated and will be removed "
                               r"after next version release. Please use '(\w+)' instead.")

    assert (re.search(pattern_with_version, str(warnings[0].message)) is not None) or\
        (re.search(pattern_without_version, str(warnings[0].message)))
