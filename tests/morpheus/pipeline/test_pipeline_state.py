#!/usr/bin/env python
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

from unittest import mock

import pytest

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.pipeline import PipelineState
from morpheus.pipeline.stage_decorator import source
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.type_aliases import DataFrameType


@source
def source_test_stage() -> int:
    for i in range(10):
        yield i


# pylint: disable=too-many-function-args


def test_build(config: Config):
    pipeline = LinearPipeline(config)
    assert pipeline.state == PipelineState.INITIALIZED
    pipeline.set_source(source_test_stage(config))
    pipeline.build()
    assert pipeline.state == PipelineState.BUILT


def test_build_after_build(config: Config):
    pipeline = LinearPipeline(config)
    assert pipeline.state == PipelineState.INITIALIZED
    pipeline.set_source(source_test_stage(config))
    pipeline.build()
    assert pipeline.state == PipelineState.BUILT
    with pytest.raises(Exception) as excinfo:
        pipeline.build()
    assert "can only be built once" in str(excinfo.value)


def test_build_without_source(config: Config):
    pipeline = LinearPipeline(config)
    assert pipeline.state == PipelineState.INITIALIZED
    with pytest.raises(Exception) as excinfo:
        pipeline.build()
    assert "must have a source stage" in str(excinfo.value)


def test_normal_run(config: Config):
    pipeline = LinearPipeline(config)
    assert pipeline.state == PipelineState.INITIALIZED
    pipeline.set_source(source_test_stage(config))
    pipeline.run()
    assert pipeline.state == PipelineState.COMPLETED


async def test_normal_build_and_start(config: Config):
    pipeline = LinearPipeline(config)
    assert pipeline.state == PipelineState.INITIALIZED
    pipeline.set_source(source_test_stage(config))
    await pipeline.build_and_start()
    assert pipeline.state == PipelineState.STARTED
    await pipeline.join()
    assert pipeline.state == PipelineState.COMPLETED


async def test_stop_after_start(config: Config):

    pipeline = LinearPipeline(config)
    assert pipeline.state == PipelineState.INITIALIZED
    pipeline.set_source(source_test_stage(config))
    await pipeline.build_and_start()
    assert pipeline.state == PipelineState.STARTED
    pipeline.stop()
    assert pipeline.state == PipelineState.STOPPED
    await pipeline.join()
    assert pipeline.state == PipelineState.COMPLETED


def test_stop_after_run(config: Config):
    pipeline = LinearPipeline(config)
    assert pipeline.state == PipelineState.INITIALIZED
    pipeline.set_source(source_test_stage(config))
    pipeline.run()
    assert pipeline.state == PipelineState.COMPLETED
    with pytest.raises(Exception) as excinfo:
        pipeline.stop()
    assert "must be running" in str(excinfo.value)


def test_stop_without_start(config: Config):

    pipeline = LinearPipeline(config)
    assert pipeline.state == PipelineState.INITIALIZED
    pipeline.set_source(source_test_stage(config))
    with pytest.raises(Exception) as excinfo:
        pipeline.stop()
    assert "must be running" in str(excinfo.value)


async def test_stop_after_stop(config: Config):

    pipeline = LinearPipeline(config)
    assert pipeline.state == PipelineState.INITIALIZED
    pipeline.set_source(source_test_stage(config))
    await pipeline.build_and_start()
    assert pipeline.state == PipelineState.STARTED
    pipeline.stop()
    assert pipeline.state == PipelineState.STOPPED
    with pytest.raises(Exception) as excinfo:
        pipeline.stop()
    assert "must be running" in str(excinfo.value)
    await pipeline.join()


async def test_join_without_start(config: Config):

    pipeline = LinearPipeline(config)
    assert pipeline.state == PipelineState.INITIALIZED
    pipeline.set_source(source_test_stage(config))
    with pytest.raises(Exception) as excinfo:
        await pipeline.join()
    assert "must be started" in str(excinfo.value)


async def test_join_after_join(config: Config):
    pipeline = LinearPipeline(config)
    assert pipeline.state == PipelineState.INITIALIZED
    pipeline.set_source(source_test_stage(config))
    await pipeline.build_and_start()
    assert pipeline.state == PipelineState.STARTED
    await pipeline.join()
    assert pipeline.state == PipelineState.COMPLETED
    await pipeline.join()
    assert pipeline.state == PipelineState.COMPLETED


@mock.patch('morpheus.stages.preprocess.deserialize_stage.DeserializeStage.join')
@mock.patch('morpheus.stages.preprocess.deserialize_stage.DeserializeStage.stop')
@mock.patch('morpheus.stages.input.in_memory_source_stage.InMemorySourceStage.join')
@mock.patch('morpheus.stages.input.in_memory_source_stage.InMemorySourceStage.stop')
def test_stage_methods_called_normal_run(mock_source_stage_stop,
                                         mock_source_stage_join,
                                         mock_deserialize_stage_stop,
                                         mock_deserialize_stage_join,
                                         config: Config,
                                         filter_probs_df: DataFrameType):
    pipeline = LinearPipeline(config)
    pipeline.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipeline.add_stage(DeserializeStage(config))
    pipeline.run()
    mock_source_stage_stop.assert_not_called()
    mock_source_stage_join.assert_called_once()
    mock_deserialize_stage_stop.assert_not_called()
    mock_deserialize_stage_join.assert_called_once()


@mock.patch('morpheus.stages.preprocess.deserialize_stage.DeserializeStage.join')
@mock.patch('morpheus.stages.preprocess.deserialize_stage.DeserializeStage.stop')
@mock.patch('morpheus.stages.input.in_memory_source_stage.InMemorySourceStage.join')
@mock.patch('morpheus.stages.input.in_memory_source_stage.InMemorySourceStage.stop')
async def test_stage_methods_called_stop_after_start(mock_source_stage_stop,
                                                     mock_source_stage_join,
                                                     mock_deserialize_stage_stop,
                                                     mock_deserialize_stage_join,
                                                     config: Config,
                                                     filter_probs_df: DataFrameType):
    pipeline = LinearPipeline(config)
    pipeline.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipeline.add_stage(DeserializeStage(config))
    await pipeline.build_and_start()
    pipeline.stop()
    await pipeline.join()
    mock_source_stage_stop.assert_called_once()
    mock_source_stage_join.assert_called_once()
    mock_deserialize_stage_stop.assert_called_once()
    mock_deserialize_stage_join.assert_called_once()


@mock.patch('morpheus.stages.preprocess.deserialize_stage.DeserializeStage.join')
@mock.patch('morpheus.stages.preprocess.deserialize_stage.DeserializeStage.stop')
@mock.patch('morpheus.stages.input.in_memory_source_stage.InMemorySourceStage.join')
@mock.patch('morpheus.stages.input.in_memory_source_stage.InMemorySourceStage.stop')
def test_stage_methods_called_stop_after_run(mock_source_stage_stop,
                                             mock_source_stage_join,
                                             mock_deserialize_stage_stop,
                                             mock_deserialize_stage_join,
                                             config: Config,
                                             filter_probs_df: DataFrameType):
    pipeline = LinearPipeline(config)
    pipeline.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipeline.add_stage(DeserializeStage(config))
    pipeline.run()
    with pytest.raises(Exception) as excinfo:
        pipeline.stop()
    assert "must be running" in str(excinfo.value)
    mock_source_stage_stop.assert_not_called()
    mock_source_stage_join.assert_called_once()
    mock_deserialize_stage_stop.assert_not_called()
    mock_deserialize_stage_join.assert_called_once()


@mock.patch('morpheus.stages.preprocess.deserialize_stage.DeserializeStage.join')
@mock.patch('morpheus.stages.preprocess.deserialize_stage.DeserializeStage.stop')
@mock.patch('morpheus.stages.input.in_memory_source_stage.InMemorySourceStage.join')
@mock.patch('morpheus.stages.input.in_memory_source_stage.InMemorySourceStage.stop')
def test_stage_methods_called_stop_without_start(mock_source_stage_stop,
                                                 mock_source_stage_join,
                                                 mock_deserialize_stage_stop,
                                                 mock_deserialize_stage_join,
                                                 config: Config,
                                                 filter_probs_df: DataFrameType):

    pipeline = LinearPipeline(config)
    assert pipeline.state == PipelineState.INITIALIZED
    pipeline.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipeline.add_stage(DeserializeStage(config))
    with pytest.raises(Exception) as excinfo:
        pipeline.stop()
    assert "must be running" in str(excinfo.value)
    mock_source_stage_stop.assert_not_called()
    mock_source_stage_join.assert_not_called()
    mock_deserialize_stage_stop.assert_not_called()
    mock_deserialize_stage_join.assert_not_called()


@mock.patch('morpheus.stages.preprocess.deserialize_stage.DeserializeStage.join')
@mock.patch('morpheus.stages.preprocess.deserialize_stage.DeserializeStage.stop')
@mock.patch('morpheus.stages.input.in_memory_source_stage.InMemorySourceStage.join')
@mock.patch('morpheus.stages.input.in_memory_source_stage.InMemorySourceStage.stop')
async def test_stage_methods_called_stop_after_stop(mock_source_stage_stop,
                                                    mock_source_stage_join,
                                                    mock_deserialize_stage_stop,
                                                    mock_deserialize_stage_join,
                                                    config: Config,
                                                    filter_probs_df: DataFrameType):
    pipeline = LinearPipeline(config)
    pipeline.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipeline.add_stage(DeserializeStage(config))
    await pipeline.build_and_start()
    pipeline.stop()
    with pytest.raises(Exception) as excinfo:
        pipeline.stop()
    assert "must be running" in str(excinfo.value)
    await pipeline.join()
    mock_source_stage_stop.assert_called_once()
    mock_source_stage_join.assert_called_once()
    mock_deserialize_stage_stop.assert_called_once()
    mock_deserialize_stage_join.assert_called_once()


@mock.patch('morpheus.stages.preprocess.deserialize_stage.DeserializeStage.join')
@mock.patch('morpheus.stages.preprocess.deserialize_stage.DeserializeStage.stop')
@mock.patch('morpheus.stages.input.in_memory_source_stage.InMemorySourceStage.join')
@mock.patch('morpheus.stages.input.in_memory_source_stage.InMemorySourceStage.stop')
async def test_stage_methods_called_join_without_start(mock_source_stage_stop,
                                                       mock_source_stage_join,
                                                       mock_deserialize_stage_stop,
                                                       mock_deserialize_stage_join,
                                                       config: Config,
                                                       filter_probs_df: DataFrameType):
    pipeline = LinearPipeline(config)
    pipeline.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipeline.add_stage(DeserializeStage(config))
    with pytest.raises(Exception) as excinfo:
        await pipeline.join()
    assert "must be started" in str(excinfo.value)
    mock_source_stage_stop.assert_not_called()
    mock_source_stage_join.assert_not_called()
    mock_deserialize_stage_stop.assert_not_called()
    mock_deserialize_stage_join.assert_not_called()


@mock.patch('morpheus.stages.preprocess.deserialize_stage.DeserializeStage.join')
@mock.patch('morpheus.stages.preprocess.deserialize_stage.DeserializeStage.stop')
@mock.patch('morpheus.stages.input.in_memory_source_stage.InMemorySourceStage.join')
@mock.patch('morpheus.stages.input.in_memory_source_stage.InMemorySourceStage.stop')
async def test_stage_methods_called_join_after_join(mock_source_stage_stop,
                                                    mock_source_stage_join,
                                                    mock_deserialize_stage_stop,
                                                    mock_deserialize_stage_join,
                                                    config: Config,
                                                    filter_probs_df: DataFrameType):
    pipeline = LinearPipeline(config)
    pipeline.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipeline.add_stage(DeserializeStage(config))
    await pipeline.build_and_start()
    await pipeline.join()
    assert pipeline.state == PipelineState.COMPLETED
    await pipeline.join()
    assert pipeline.state == PipelineState.COMPLETED
    mock_source_stage_stop.assert_not_called()
    mock_source_stage_join.assert_called_once()
    mock_deserialize_stage_stop.assert_not_called()
    mock_deserialize_stage_join.assert_called_once()
