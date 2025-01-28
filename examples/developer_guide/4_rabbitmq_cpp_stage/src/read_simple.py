#!/usr/bin/env python3
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

import logging
import os

import click
from rabbitmq_cpp_stage.rabbitmq_source_stage import RabbitMQSourceStage

from morpheus.common import FileTypes
from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.utils.logger import configure_logging


@click.command()
@click.option('--use_cpu_only', default=False, type=bool, is_flag=True, help=("Whether or not to run in CPU only mode"))
@click.option(
    "--num_threads",
    default=len(os.sched_getaffinity(0)),
    type=click.IntRange(min=1),
    help="Number of internal pipeline threads to use",
)
def run_pipeline(use_cpu_only: bool, num_threads: int):
    # Enable the Morpheus logger
    configure_logging(log_level=logging.DEBUG)

    config = Config()
    config.execution_mode = ExecutionMode.CPU if use_cpu_only else ExecutionMode.GPU
    config.num_threads = num_threads

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    # Set source stage
    pipeline.set_source(RabbitMQSourceStage(config, host='localhost', exchange='logs'))

    # Add monitor to record the performance of our new stages
    pipeline.add_stage(MonitorStage(config))

    # Write the to the output file
    pipeline.add_stage(WriteToFileStage(config, filename='results.json', file_type=FileTypes.JSON, overwrite=True))

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
