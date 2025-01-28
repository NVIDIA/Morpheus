#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import queue
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import cudf

from _utils import assert_results
from _utils import mk_async_infer
from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.config import ConfigFIL
from morpheus.config import PipelineModes
from morpheus.pipeline import LinearPipeline
from morpheus.stages.inference.triton_inference_stage import ProducerConsumerQueue
from morpheus.stages.inference.triton_inference_stage import ResourcePool
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceWorker
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.postprocess.add_scores_stage import AddScoresStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_fil_stage import PreprocessFILStage

MODEL_MAX_BATCH_SIZE = 1024


def test_resource_pool():
    create_fn = mock.MagicMock()

    # If called a third time this will raise a StopIteration exception
    create_fn.side_effect = range(2)

    pool = ResourcePool[int](create_fn=create_fn, max_size=2)

    assert pool._queue.qsize() == 0

    # Check for normal allocation
    assert pool.borrow_obj() == 0
    assert pool._queue.qsize() == 0
    assert pool.added_count == 1
    create_fn.assert_called_once()

    assert pool.borrow_obj() == 1
    assert pool._queue.qsize() == 0
    assert pool.added_count == 2
    assert create_fn.call_count == 2

    pool.return_obj(0)
    assert pool._queue.qsize() == 1
    pool.return_obj(1)
    assert pool._queue.qsize() == 2

    assert pool.borrow_obj() == 0
    assert pool._queue.qsize() == 1
    assert pool._added_count == 2
    assert create_fn.call_count == 2

    assert pool.borrow_obj() == 1
    assert pool._queue.qsize() == 0
    assert pool._added_count == 2
    assert create_fn.call_count == 2


def test_resource_pool_overallocate():
    create_fn = mock.MagicMock()

    # If called a third time this will raise a StopIteration exception
    create_fn.side_effect = range(5)

    pool = ResourcePool[int](create_fn=create_fn, max_size=2)

    assert pool.borrow_obj() == 0
    assert pool.borrow_obj() == 1

    with pytest.raises(queue.Empty):
        pool.borrow_obj(timeout=0)


def test_resource_pool_large_count():
    create_fn = mock.MagicMock()
    create_fn.side_effect = range(10000)

    pool = ResourcePool[int](create_fn=create_fn, max_size=10000)

    for _ in range(10000):
        pool.borrow_obj(timeout=0)

    assert pool._queue.qsize() == 0
    assert create_fn.call_count == 10000


def test_resource_pool_create_raises_error():
    create_fn = mock.MagicMock()
    create_fn.side_effect = (10, RuntimeError, 20)

    pool = ResourcePool[int](create_fn=create_fn, max_size=10)

    assert pool.borrow_obj() == 10

    with pytest.raises(RuntimeError):
        pool.borrow_obj()

    assert pool.borrow_obj() == 20


@pytest.mark.gpu_mode
@pytest.mark.parametrize("pipeline_mode", list(PipelineModes))
def test_stage_constructor_worker_class(config: Config, pipeline_mode: PipelineModes):
    config.mode = pipeline_mode
    stage = TritonInferenceStage(config, model_name='test', server_url='test:0000', use_shared_memory=True)
    worker = stage._get_inference_worker(ProducerConsumerQueue())
    assert isinstance(worker, TritonInferenceWorker)


@pytest.mark.gpu_mode
@pytest.mark.parametrize("pipeline_mode", list(PipelineModes))
@pytest.mark.parametrize("needs_logits", [True, False, None])
def test_stage_get_inference_worker(config: Config, pipeline_mode: PipelineModes, needs_logits: bool | None):
    if needs_logits is None:
        expexted_needs_logits = (pipeline_mode == PipelineModes.NLP)
    else:
        expexted_needs_logits = needs_logits

    config.mode = pipeline_mode

    stage = TritonInferenceStage(config,
                                 model_name='test',
                                 server_url='test:0000',
                                 needs_logits=needs_logits,
                                 use_shared_memory=True)

    worker = stage._get_inference_worker(ProducerConsumerQueue())
    assert isinstance(worker, TritonInferenceWorker)
    assert worker.needs_logits == expexted_needs_logits


@pytest.mark.slow
@pytest.mark.gpu_mode
@pytest.mark.parametrize('num_records', [10])
@mock.patch('tritonclient.grpc.InferenceServerClient')
def test_triton_stage_pipe(mock_triton_client, config, num_records):
    mock_metadata = {
        "inputs": [{
            'name': 'input__0', 'datatype': 'FP32', "shape": [-1, 1]
        }],
        "outputs": [{
            'name': 'output__0', 'datatype': 'FP32', 'shape': ['-1', '1']
        }]
    }
    mock_model_config = {"config": {"max_batch_size": MODEL_MAX_BATCH_SIZE}}

    input_df = pd.DataFrame(data={'v': (i * 2 for i in range(num_records))})
    expected_df = pd.DataFrame(data={'v': input_df['v'], 'score_test': input_df['v']})

    mock_triton_client.return_value = mock_triton_client
    mock_triton_client.is_server_live.return_value = True
    mock_triton_client.is_server_ready.return_value = True
    mock_triton_client.is_model_ready.return_value = True
    mock_triton_client.get_model_metadata.return_value = mock_metadata
    mock_triton_client.get_model_config.return_value = mock_model_config

    inf_results = np.split(input_df.values, range(MODEL_MAX_BATCH_SIZE, len(input_df), MODEL_MAX_BATCH_SIZE))

    async_infer = mk_async_infer(inf_results)
    mock_triton_client.async_infer.side_effect = async_infer

    config.mode = PipelineModes.FIL
    config.class_labels = ["test"]
    config.model_max_batch_size = MODEL_MAX_BATCH_SIZE
    config.pipeline_batch_size = 1024
    config.feature_length = 1
    config.edge_buffer_size = 128
    config.num_threads = 1

    config.fil = ConfigFIL()
    config.fil.feature_columns = ['v']

    pipe_cm = LinearPipeline(config)
    pipe_cm.set_source(InMemorySourceStage(config, [cudf.DataFrame(input_df)]))
    pipe_cm.add_stage(DeserializeStage(config))
    pipe_cm.add_stage(PreprocessFILStage(config))
    pipe_cm.add_stage(
        # Intentionally using use_shared_memory=True as this is the only way to use the Python impl
        TritonInferenceStage(config,
                             model_name='abp-nvsmi-xgb',
                             server_url='test:0000',
                             force_convert_inputs=True,
                             use_shared_memory=True))
    pipe_cm.add_stage(AddScoresStage(config, prefix="score_", probs_type=TypeId.FLOAT64))
    pipe_cm.add_stage(SerializeStage(config))
    comp_stage = pipe_cm.add_stage(CompareDataFrameStage(config, expected_df))

    pipe_cm.run()

    assert_results(comp_stage.get_results())
