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

import logging

import pytest
from static_message_source import StaticMessageSource

import cudf

from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.logger import configure_logging


def build_and_run_pipeline(config: Config, df: cudf.DataFrame):
    # Pipeline
    pipeline = LinearPipeline(config)

    # Necessary pieces we'd rather not benchmark if we could avoid it.
    pipeline.set_source(StaticMessageSource(config, df))
    pipeline.add_stage(DeserializeStage(config))

    # Stage we want to benchmark
    pipeline.add_stage(SerializeStage(config))

    pipeline.build()
    pipeline.run()


@pytest.mark.benchmark
@pytest.mark.parametrize("num_messages", [1, 100, 10000])
def test_monitor_stage(benchmark, num_messages):

    # Test Data

    df = cudf.DataFrame({"value": list(range(0, num_messages))})

    # Configuration

    configure_logging(log_level=logging.INFO)

    config = Config()
    CppConfig.set_should_use_cpp(True)
    config.mode = PipelineModes.OTHER
    config.use_dask = False

    config.num_threads = 1
    config.pipeline_batch_size = 64
    config.model_max_batch_size = 64
    config.feature_length = 32

    config.class_labels = ["probs"]
    config.edge_buffer_size = 4

    # would prefer to benchmark just pipeline.run, but it asserts when called multiple times
    benchmark(build_and_run_pipeline, config, df)
