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
import os

import click

logger = logging.getLogger(__name__)


@click.group(name=__name__)
def run():
    pass


@run.command(help="Runs a simple finite pipeline with a single execution of a LangChain agent from a fixed input")
@click.option('--use_cpu_only', default=False, type=bool, is_flag=True, help="Run in CPU only mode")
@click.option(
    "--num_threads",
    default=len(os.sched_getaffinity(0)),
    type=click.IntRange(min=1),
    help="Number of internal pipeline threads to use",
)
@click.option(
    "--pipeline_batch_size",
    default=1024,
    type=click.IntRange(min=1),
    help=("Internal batch size for the pipeline. Can be much larger than the model batch size. "
          "Also used for Kafka consumers"),
)
@click.option(
    "--model_max_batch_size",
    default=64,
    type=click.IntRange(min=1),
    help="Max batch size to use for the model",
)
@click.option(
    "--model_name",
    required=True,
    type=str,
    default='gpt-3.5-turbo-instruct',
    help="The name of the model to use in OpenAI",
)
@click.option(
    "--repeat_count",
    default=1,
    type=click.IntRange(min=1),
    help="Number of times to repeat the input query. Useful for testing performance.",
)
def simple(**kwargs):

    from .simple_pipeline import pipeline as _pipeline

    return _pipeline(**kwargs)


@run.command(help="Runs a pipeline LangChain agents which pulls inputs from a Kafka message bus")
@click.option(
    "--num_threads",
    default=len(os.sched_getaffinity(0)),
    type=click.IntRange(min=1),
    help="Number of internal pipeline threads to use",
)
@click.option(
    "--pipeline_batch_size",
    default=1024,
    type=click.IntRange(min=1),
    help=("Internal batch size for the pipeline. Can be much larger than the model batch size. "
          "Also used for Kafka consumers"),
)
@click.option(
    "--model_max_batch_size",
    default=64,
    type=click.IntRange(min=1),
    help="Max batch size to use for the model",
)
@click.option(
    "--model_name",
    required=True,
    type=str,
    default='gpt-3.5-turbo-instruct',
    help="The name of the model to use in OpenAI",
)
@click.option(
    "--bootstrap_servers",
    required=True,
    type=str,
    default='auto',
    help="The Kafka bootstrap servers to connect to",
)
@click.option(
    "--topic",
    required=True,
    type=str,
    default='input',
    help="The Kafka topic to listen to for input messages",
)
def kafka(**kwargs):

    from .kafka_pipeline import pipeline as _pipeline

    return _pipeline(**kwargs)
