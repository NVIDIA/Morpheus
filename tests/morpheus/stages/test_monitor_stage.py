#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import inspect
import logging
import os
import threading
import typing
from unittest import mock

import fsspec
import pytest

import cudf

from _utils import TEST_DIRS
from _utils.stages.record_thread_id_stage import RecordThreadIdStage
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.utils.logger import set_log_level


def test_constructor(config: Config):
    # Intentionally not making assumptions about the defaults other than they exist
    # and still create a valid stage.
    stage = MonitorStage(config, log_level=logging.WARNING)
    assert stage.name == "monitor"

    # Just ensure that we get a valid non-empty tuple
    accepted_types = stage.accepted_types()
    assert isinstance(accepted_types, tuple)
    assert len(accepted_types) > 0

    def two_x(x):
        return x * 2

    stage = MonitorStage(config, description="Test Description", smoothing=0.7, unit='units', determine_count_fn=two_x)
    assert stage._mc._description == "Test Description"
    assert stage._mc._smoothing == 0.7
    assert stage._mc._unit == "units"
    assert stage._mc._determine_count_fn is two_x


@pytest.mark.cpu_mode
@mock.patch('morpheus.controllers.monitor_controller.MorpheusTqdm')
def test_start_async(mock_morph_tqdm: mock.MagicMock, config: Config):
    mock_morph_tqdm.return_value = mock_morph_tqdm

    stage = MonitorStage(config, delayed_start=False, log_level=logging.WARNING)
    assert stage._mc._progress is None

    asyncio.run(stage.start_async())
    mock_morph_tqdm.assert_called_once()
    mock_morph_tqdm.reset.assert_called_once()
    assert stage._mc._progress is mock_morph_tqdm


@pytest.mark.cpu_mode
@mock.patch('morpheus.controllers.monitor_controller.MorpheusTqdm')
async def test_join(mock_morph_tqdm: mock.MagicMock, config: Config):
    mock_morph_tqdm.return_value = mock_morph_tqdm

    stage = MonitorStage(config, delayed_start=False, log_level=logging.WARNING)
    assert stage._mc._progress is None

    # Calling join is a noop if we are stopped
    await stage.join()
    mock_morph_tqdm.assert_not_called()

    await stage.start_async()
    await stage.join()
    mock_morph_tqdm.close.assert_called_once()


@pytest.mark.cpu_mode
@mock.patch('morpheus.controllers.monitor_controller.MorpheusTqdm')
def test_refresh(mock_morph_tqdm: mock.MagicMock, config: Config):
    mock_morph_tqdm.return_value = mock_morph_tqdm

    stage = MonitorStage(config, delayed_start=False, log_level=logging.WARNING)
    assert stage._mc._progress is None

    asyncio.run(stage.start_async())
    stage._mc.refresh_progress(None)
    mock_morph_tqdm.refresh.assert_called_once()


@pytest.mark.parametrize('value,expected_fn,expected',
                         [
                             (None, False, None),
                             ([], False, None),
                             (['s'], True, 1),
                             ('s', True, 1),
                             ('test', True, 1),
                             (cudf.DataFrame(), True, 0),
                             (cudf.DataFrame(range(12), columns=["test"]), True, 12),
                             ({}, True, 0),
                             (tuple(), True, 0),
                             (set(), True, 0),
                             (fsspec.open_files(os.path.join(TEST_DIRS.tests_data_dir, 'filter_probs.csv')), True, 1),
                         ])
def test_auto_count_fn(config: Config, value: typing.Any, expected_fn: bool, expected: typing.Union[int, None]):
    stage = MonitorStage(config, log_level=logging.WARNING)

    auto_fn = stage._mc.auto_count_fn(value)
    if expected_fn:
        assert callable(auto_fn)
        assert auto_fn(value) == expected
    else:
        assert auto_fn is None


@pytest.mark.parametrize('value', [1, [1], [2, 0]])
def test_auto_count_fn_not_impl(config: Config, value: typing.Any):
    stage = MonitorStage(config, log_level=logging.WARNING)

    with pytest.raises(NotImplementedError):
        stage._mc.auto_count_fn(value)


@mock.patch('morpheus.controllers.monitor_controller.MorpheusTqdm')
def test_progress_sink(mock_morph_tqdm: mock.MagicMock, config: Config):
    mock_morph_tqdm.return_value = mock_morph_tqdm

    stage = MonitorStage(config, log_level=logging.WARNING)
    asyncio.run(stage.start_async())

    stage._mc.progress_sink(None)
    assert stage._mc._determine_count_fn is None
    mock_morph_tqdm.update.assert_not_called()

    cm = ControlMessage()
    cm.payload(MessageMeta(df=cudf.DataFrame(range(12), columns=["test"])))
    stage._mc.progress_sink(cm)
    assert inspect.isfunction(stage._mc._determine_count_fn)
    mock_morph_tqdm.update.assert_called_once_with(n=12)


@pytest.mark.cpu_mode
@pytest.mark.usefixtures("reset_loglevel")
@pytest.mark.parametrize('log_level', [logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG])
@mock.patch('morpheus.stages.general.monitor_stage.MonitorController.sink_on_completed', autospec=True)
@mock.patch('morpheus.stages.general.monitor_stage.MonitorController.progress_sink', autospec=True)
def test_log_level(mock_progress_sink: mock.MagicMock,
                   mock_sink_on_completed: mock.MagicMock,
                   config: Config,
                   log_level: int):
    """
    Test ensures the monitor stage doesn't add itself to the MRC pipeline if not configured for the current log-level
    """
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")

    set_log_level(log_level)
    monitor_stage_level = logging.INFO

    should_be_included = (log_level <= monitor_stage_level)

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file))
    pipe.add_stage(MonitorStage(config, log_level=monitor_stage_level))
    pipe.run()

    expected_call_count = 1 if should_be_included else 0
    assert mock_progress_sink.call_count == expected_call_count
    assert mock_sink_on_completed.call_count == expected_call_count


@pytest.mark.cpu_mode
def test_thread(config: Config, morpheus_log_level: int):
    """
    Test ensures the monitor stage executes on the same thread as the parent stage
    """
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")

    monitor_thread_id = None

    # Create a dummy count function where we can save the thread id from the monitor stage
    def fake_determine_count_fn(x):
        nonlocal monitor_thread_id

        monitor_thread_id = threading.current_thread().ident

        return x.count

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file))
    dummy_stage = pipe.add_stage(RecordThreadIdStage(config))
    pipe.add_stage(MonitorStage(config, determine_count_fn=fake_determine_count_fn, log_level=morpheus_log_level))
    pipe.run()

    # Check that the thread ids are the same
    assert monitor_thread_id is not None
    assert dummy_stage.thread_id == monitor_thread_id
