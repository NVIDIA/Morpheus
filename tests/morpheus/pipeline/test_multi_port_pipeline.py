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

import cudf

from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.pipeline.pipeline import Pipeline
from morpheus.stages.general.multi_port_modules_stage import MultiPortModulesStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage

# When segment modules are imported, they're added to the module registry.
# To avoid flake8 warnings about unused code, the noqa flag is used during import.
import _modules.multiplexer  # noqa: F401 # pylint: disable=unused-import # isort:skip


def _run_pipeline(config: Config, source_df: cudf.DataFrame, module_conf: dict,
                  stage_input_ports: list[str]) -> InMemorySinkStage:
    pipe = Pipeline(config)

    mux_stage = pipe.add_stage(
        MultiPortModulesStage(config, module_conf, input_ports=stage_input_ports, output_ports=["output"]))

    for x in range(len(stage_input_ports)):
        source_stage = pipe.add_stage(InMemorySourceStage(config, [source_df.copy(deep=True)]))
        pipe.add_edge(source_stage, mux_stage.input_ports[x])

    sink_stage = pipe.add_stage(InMemorySinkStage(config))

    pipe.add_edge(mux_stage, sink_stage)

    pipe.run()

    return sink_stage


@pytest.mark.parametrize("source_count, expected_count", [(1, 1), (2, 2), (3, 3)])
def test_multi_port_pipeline(config: Config, dataset_cudf: DatasetManager, source_count, expected_count):

    filter_probs_df = dataset_cudf["filter_probs.csv"]

    input_ports = [f"input_{x}" for x in range(source_count)]

    multiplexer_module_conf = {
        "module_id": "multiplexer",
        "namespace": "morpheus_test",
        "module_name": "multiplexer",
        "input_ports": input_ports,
        "output_port": "output"
    }

    sink_stage = _run_pipeline(config=config,
                               source_df=filter_probs_df,
                               module_conf=multiplexer_module_conf,
                               stage_input_ports=input_ports)
    assert len(sink_stage.get_messages()) == expected_count


def test_multi_port_pipeline_mis_config(config: Config, dataset_cudf: DatasetManager):
    config_input_ports = ["input_0", "input_1"]
    stage_input_ports = ["input_0", "input_1", "input_2"]

    filter_probs_df = dataset_cudf["filter_probs.csv"]

    multiplexer_module_conf = {
        "module_id": "multiplexer",
        "namespace": "morpheus_test",
        "module_name": "multiplexer",
        "input_ports": config_input_ports,
        "output_port": "output"
    }

    with pytest.raises(ValueError):
        _run_pipeline(config=config,
                      source_df=filter_probs_df,
                      module_conf=multiplexer_module_conf,
                      stage_input_ports=stage_input_ports)
