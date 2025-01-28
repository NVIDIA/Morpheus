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
    "--embedding_size",
    default=384,
    type=click.IntRange(min=1),
    help="The output size of the embedding calculation. Depends on the model supplied by --model_name",
)
@click.option(
    "--model_name",
    required=True,
    type=str,
    default='gpt-43b-002',
    help="The name of the large language model that is deployed on Triton server",
)
@click.option(
    "--vdb_resource_name",
    required=True,
    type=str,
    default='RSS',
    help="The name resource on the Vector Database where the embeddings are stored",
)
@click.option(
    "--repeat_count",
    default=1,
    type=click.IntRange(min=1),
    help="Number of times to repeat the input query. Useful for testing performance.",
)
@click.option(
    "--llm_service",
    default="NemoLLM",
    type=click.Choice(['NemoLLM', 'OpenAI'], case_sensitive=False),
    help="LLM service to issue requests to, should be used in conjunction with --model_name.",
)
@click.option(
    "--question",
    type=str,
    multiple=True,
    default=["What are some new attacks discovered in the cyber security industry?"] * 5,
    help="The question to answer with the RAG pipeline. Specify multiple times to answer multiple questions at once.",
)
def pipeline(**kwargs):

    from .standalone_pipeline import standalone

    return standalone(**kwargs)
