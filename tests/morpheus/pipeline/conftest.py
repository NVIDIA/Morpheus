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

import pytest

from _utils.dataset_manager import DatasetManager
from _utils.stages.in_memory_multi_source_stage import InMemoryMultiSourceStage
from _utils.stages.multi_port_pass_thru import MultiPortPassThruStage
from _utils.stages.split_stage import SplitStage
from morpheus.config import Config
from morpheus.pipeline import Pipeline
from morpheus.pipeline.source_stage import SourceStage
from morpheus.pipeline.stage_base import StageBase
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage


@pytest.fixture(name="in_mem_source_stage")
def in_mem_source_stage_fixture(config: Config, dataset_cudf: DatasetManager):
    df = dataset_cudf['filter_probs.csv']
    yield InMemorySourceStage(config, [df])


@pytest.fixture(name="in_mem_multi_source_stage")
def in_mem_multi_source_stage_fixture(config: Config):
    data = [[1, 2, 3], ["a", "b", "c"], [1.1, 2.2, 3.3]]
    yield InMemoryMultiSourceStage(config, data=data)


def _build_ports(config: Config, source_stage: SourceStage, stage: StageBase):
    pipe = Pipeline(config)
    pipe.add_stage(source_stage)
    pipe.add_stage(stage)

    for (port_idx, output_port) in enumerate(source_stage.output_ports):
        pipe.add_edge(output_port, stage.input_ports[port_idx])

    pipe.build()


@pytest.fixture(name="stage")
def stage_fixture(config: Config, in_mem_source_stage: InMemorySourceStage):
    stage = DeserializeStage(config)
    _build_ports(config=config, source_stage=in_mem_source_stage, stage=stage)

    yield stage


@pytest.fixture(name="split_stage")
def split_stage_fixture(config: Config, in_mem_source_stage: InMemorySourceStage):
    stage = SplitStage(config)
    _build_ports(config=config, source_stage=in_mem_source_stage, stage=stage)

    yield stage


@pytest.fixture(name="multi_pass_thru_stage")
def multi_pass_thru_stage_fixture(config: Config, in_mem_multi_source_stage: InMemoryMultiSourceStage):
    stage = MultiPortPassThruStage(config, num_ports=3)
    _build_ports(config=config, source_stage=in_mem_multi_source_stage, stage=stage)

    yield stage
