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


@run.command()
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
    "--repeat_count",
    default=1,
    type=click.IntRange(min=1),
    help="Number of times to repeat the input query. Useful for testing performance.",
)
@click.option("--llm_service",
              default="NemoLLM",
              type=click.Choice(['NemoLLM', 'OpenAI'], case_sensitive=False),
              help="LLM service to issue requests to.")
@click.option(
    "--input_file",
    type=click.Path(exists=True, readable=True),
    default=None,
    required=False,
    help="Input to read country names from, if undefined an in-memory DataFrame of ten countris will be used.")
@click.option(
    "--shuffle",
    is_flag=True,
    default=False,
    help=("Random shuffle order of country names."),
)
def pipeline(**kwargs):
    from .pipeline import pipeline as _pipeline

    return _pipeline(**kwargs)
