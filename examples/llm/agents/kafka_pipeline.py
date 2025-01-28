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
import time

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.kafka_source_stage import KafkaSourceStage

from .common import build_common_pipeline

logger = logging.getLogger(__name__)


def pipeline(num_threads: int,
             pipeline_batch_size: int,
             model_max_batch_size: int,
             model_name: str,
             bootstrap_servers: str,
             topic: str) -> float:
    config = Config()
    config.mode = PipelineModes.OTHER

    # Below properties are specified by the command line
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.mode = PipelineModes.NLP
    config.edge_buffer_size = 128

    completion_task = {"task_type": "completion", "task_dict": {"input_keys": ["question"], }}

    pipe = LinearPipeline(config)

    pipe.set_source(KafkaSourceStage(config, bootstrap_servers=bootstrap_servers, input_topic=[topic]))

    sink = build_common_pipeline(config=config, pipe=pipe, task_payload=completion_task, model_name=model_name)

    start_time = time.time()

    logger.info("Pipeline running. Waiting for input from Kafka...")

    pipe.run()

    logger.info("Pipeline complete. Received %s responses", len(sink.get_messages()))

    return start_time
