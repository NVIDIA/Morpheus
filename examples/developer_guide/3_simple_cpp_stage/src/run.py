#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from simple_cpp_stage.pass_thru import PassThruStage

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.logger import configure_logging


def run_pipeline():
    # Enable the Morpheus logger
    configure_logging(log_level=logging.DEBUG)

    root_dir = os.environ['MORPHEUS_ROOT']
    input_file = os.path.join(root_dir, 'examples/data/email_with_addresses.jsonlines')

    config = Config()

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    # Set source stage
    pipeline.set_source(FileSourceStage(config, filename=input_file, iterative=False))

    # Add a PassThruStage where the input type is MessageMeta
    pipeline.add_stage(PassThruStage(config))

    pipeline.add_stage(DeserializeStage(config))

    # Add a PassThruStage where the input type is ControlMessage
    pipeline.add_stage(PassThruStage(config))

    # Add monitor to record the performance of our new stage
    pipeline.add_stage(MonitorStage(config))

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
