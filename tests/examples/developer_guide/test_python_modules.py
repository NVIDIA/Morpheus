# Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

import os
import types

import pytest

import cudf

from _utils import TEST_DIRS
from _utils import assert_results
from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage

EXAMPLES_DIR = os.path.join(TEST_DIRS.examples_dir, "developer_guide", "7_python_modules")


@pytest.mark.import_mod([
    os.path.join(EXAMPLES_DIR, "my_test_compound_module.py"),
    os.path.join(EXAMPLES_DIR, "my_test_module.py"),
    os.path.join(EXAMPLES_DIR, "my_test_module_consumer.py"),
    os.path.join(EXAMPLES_DIR, "my_compound_module_consumer_stage.py"),
    os.path.join(EXAMPLES_DIR, "my_test_module_consumer_stage.py")
])
def test_pipeline(config: Config, import_mod: list[types.ModuleType], morpheus_log_level: int):
    my_compound_module_consumer_stage = import_mod[-2]
    my_test_module_consumer_stage = import_mod[-1]

    input_df = cudf.DataFrame({"data": [1.2, 2.3, 3.4, 4.5, 5.6]})
    expected_df = cudf.DataFrame({"data": [4.32, 15.87, 34.68, 60.75, 94.08]})

    pipeline = LinearPipeline(config)
    pipeline.set_source(InMemorySourceStage(config, [input_df]))

    pipeline.add_stage(
        LinearModulesStage(
            config,
            module_config={
                "module_id": "my_test_module",
                "namespace": "my_module_namespace",
                "module_name": "module_instance_name",  # ... other module config params...
            },
            input_port_name="input_0",
            output_port_name="output_0"))

    pipeline.add_stage(
        LinearModulesStage(
            config,
            module_config={
                "module_id": "my_test_module_consumer",
                "namespace": "my_module_namespace",
                "module_name": "my_test_module_consumer",  # ... other module config params...
            },
            input_port_name="input_0",
            output_port_name="output_0"))

    pipeline.add_stage(my_test_module_consumer_stage.MyPassthroughModuleWrapper(config))
    pipeline.add_stage(my_compound_module_consumer_stage.MyCompoundOpModuleWrapper(config))
    pipeline.add_stage(MonitorStage(config, log_level=morpheus_log_level))
    comp_stage = pipeline.add_stage(CompareDataFrameStage(config, expected_df))

    pipeline.run()

    assert_results(comp_stage.get_results())
