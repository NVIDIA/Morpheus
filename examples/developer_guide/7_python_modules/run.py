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

import logging

# pylint: disable=unused-import
import my_test_compound_module  # noqa: F401
import my_test_module  # noqa: F401
import my_test_module_consumer  # noqa: F401
# pylint: enable=unused-import
from my_compound_module_consumer_stage import MyCompoundOpModuleWrapper
from my_test_module_consumer_stage import MyPassthroughModuleWrapper

import cudf

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils import concat_df
from morpheus.utils.logger import configure_logging

# Configure a logger under the morpheus namespace
logger = logging.getLogger(f"morpheus.{__file__}")


def run_pipeline():
    # Enable the Morpheus logger
    configure_logging(log_level=logging.DEBUG)

    input_df = cudf.DataFrame({"data": [1.2, 2.3, 3.4, 4.5, 5.6]})

    config = Config()  # Morpheus config
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

    pipeline.add_stage(MyPassthroughModuleWrapper(config))
    pipeline.add_stage(MyCompoundOpModuleWrapper(config))
    pipeline.add_stage(MonitorStage(config))
    sink = pipeline.add_stage(InMemorySinkStage(config))

    pipeline.run()

    results = concat_df.concat_dataframes(sink.get_messages())
    logger.info("Results:\n%s", results)


if __name__ == "__main__":
    run_pipeline()
