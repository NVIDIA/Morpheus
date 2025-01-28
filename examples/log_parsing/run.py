# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
# pylint: disable=no-name-in-module
from inference import LogParsingInferenceStage
from postprocessing import LogParsingPostProcessingStage

from morpheus.cli.utils import MorpheusRelativePath
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage
from morpheus.utils.logger import configure_logging


@click.command()
@click.option(
    "--num_threads",
    default=len(os.sched_getaffinity(0)),
    type=click.IntRange(min=1),
    help="Number of internal pipeline threads to use.",
)
@click.option(
    "--pipeline_batch_size",
    default=1024,
    type=click.IntRange(min=1),
    help=("Internal batch size for the pipeline. Can be much larger than the model batch size. "
          "Also used for Kafka consumers."),
)
@click.option(
    "--model_max_batch_size",
    default=32,
    type=click.IntRange(min=1),
    help="Max batch size to use for the model.",
)
@click.option(
    "--input_file",
    type=click.Path(exists=True, readable=True),
    required=True,
    help="Input filepath.",
)
@click.option(
    "--output_file",
    default=".tmp/output/log-parsing-output.jsonlines",
    help="The path to the file where the inference output will be saved.",
)
@click.option('--model_vocab_hash_file',
              required=True,
              type=MorpheusRelativePath(exists=True, dir_okay=False),
              help="Model vocab hash file to use for pre-processing.")
@click.option('--model_vocab_file',
              required=True,
              type=MorpheusRelativePath(exists=True, dir_okay=False),
              help="Model vocab file to use for post-processing.")
@click.option("--model_seq_length",
              default=256,
              type=click.IntRange(min=1),
              help="Sequence length to use for the model.")
@click.option(
    "--model_name",
    required=True,
    help="The name of the model that is deployed on Tritonserver.",
)
@click.option("--model_config_file", required=True, help="Model config file.")
@click.option("--server_url", required=True, help="Tritonserver url.", default="localhost:8001")
def run_pipeline(
    num_threads,
    pipeline_batch_size,
    model_max_batch_size,
    input_file,
    output_file,
    model_vocab_hash_file,
    model_vocab_file,
    model_seq_length,
    model_name,
    model_config_file,
    server_url,
):

    # Enable the default logger.
    configure_logging(log_level=logging.INFO)

    config = Config()
    config.mode = PipelineModes.NLP
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.feature_length = model_seq_length

    # Create a pipeline object.
    pipeline = LinearPipeline(config)

    # Add a source stage.
    # In this stage, messages were loaded from a file.
    pipeline.set_source(FileSourceStage(config, filename=input_file, iterative=False, repeat=1))

    # Add a deserialize stage.
    # At this stage, messages were logically partitioned based on the 'pipeline_batch_size'.
    pipeline.add_stage(DeserializeStage(config))

    # Add a preprocessing stage.
    # This stage preprocess the rows in the Dataframe.
    pipeline.add_stage(
        PreprocessNLPStage(config,
                           vocab_hash_file=model_vocab_hash_file,
                           truncation=False,
                           do_lower_case=False,
                           stride=64,
                           add_special_tokens=False,
                           column="raw"))

    # Add a monitor stage.
    # This stage logs the metrics (msg/sec) from the above stage.
    pipeline.add_stage(MonitorStage(config, description="Preprocessing rate"))

    # Add a inference stage.
    # This stage sends inference requests to the Tritonserver and captures the response.
    pipeline.add_stage(
        LogParsingInferenceStage(config, model_name=model_name, server_url=server_url, force_convert_inputs=True))

    # Add a monitor stage.
    # This stage logs the metrics (msg/sec) from the above stage.
    pipeline.add_stage(MonitorStage(config, description="Inference rate", unit="inf"))

    # Add a podt processing stage.
    # This stage does post-processing on the inference response.
    pipeline.add_stage(
        LogParsingPostProcessingStage(config, vocab_path=model_vocab_file, model_config_path=model_config_file))

    # Add a write file stage.
    # This stage writes all messages to a file.
    pipeline.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))

    # Add a monitor stage.
    # This stage logs the metrics (msg/sec) from the above stage.
    pipeline.add_stage(MonitorStage(config, description="Postprocessing rate"))

    # Run the pipeline.
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
