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
from stages.classification_stage import ClassificationStage
from stages.graph_construction_stage import FraudGraphConstructionStage
from stages.graph_sage_stage import GraphSAGEStage

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.logger import configure_logging

CUR_DIR = os.path.dirname(__file__)


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
    "--model_fea_length",
    default=70,
    type=click.IntRange(min=1),
    help="Features length to use for the model.",
)
@click.option(
    "--input_file",
    type=click.Path(exists=True, readable=True, dir_okay=False),
    default=os.path.join(CUR_DIR, "validation.csv"),
    required=True,
    help="Input data filepath.",
)
@click.option(
    "--training_file",
    type=click.Path(exists=True, readable=True, dir_okay=False),
    default=os.path.join(CUR_DIR, "training.csv"),
    required=True,
    help="Training data filepath.",
)
@click.option(
    "--model_dir",
    type=click.Path(exists=True, readable=True, file_okay=False, dir_okay=True),
    default=os.path.join(CUR_DIR, "model"),
    required=True,
    help="Path to trained Hinsage & XGB models.",
)
@click.option(
    "--output_file",
    type=click.Path(dir_okay=False),
    default=".tmp/output/gnn_fraud_detection_output.csv",
    help="The path to the file where the inference output will be saved.",
)
def run_pipeline(num_threads,
                 pipeline_batch_size,
                 model_max_batch_size,
                 model_fea_length,
                 input_file,
                 training_file,
                 model_dir,
                 output_file):
    # Enable the default logger.
    configure_logging(log_level=logging.INFO)

    # Its necessary to get the global config object and configure it for FIL mode.
    config = Config()
    config.mode = PipelineModes.OTHER

    # Below properties are specified by the command line.
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.feature_length = model_fea_length

    config.class_labels = ["probs"]
    config.edge_buffer_size = 4

    # Create a linear pipeline object.
    pipeline = LinearPipeline(config)

    # Set source stage.
    # In this stage, messages were loaded from a file.
    pipeline.set_source(FileSourceStage(
        config,
        filename=input_file,
        filter_null=False,
    ))

    # Add a deserialize stage.
    # At this stage, messages were logically partitioned based on the 'pipeline_batch_size'.
    pipeline.add_stage(DeserializeStage(config))

    # Add the graph construction stage.
    pipeline.add_stage(FraudGraphConstructionStage(config, training_file))
    pipeline.add_stage(MonitorStage(config, description="Graph construction rate"))

    # Add a sage inference stage.
    pipeline.add_stage(GraphSAGEStage(config, model_dir))
    pipeline.add_stage(MonitorStage(config, description="Inference rate"))

    # Add classification stage.
    # This stage adds detected classifications to each message.
    pipeline.add_stage(ClassificationStage(config, os.path.join(model_dir, "xgb.pt")))
    pipeline.add_stage(MonitorStage(config, description="Add classification rate"))

    # Add a serialize stage.
    # This stage includes & excludes columns from messages.
    pipeline.add_stage(SerializeStage(config))
    pipeline.add_stage(MonitorStage(config, description="Serialize rate"))

    # Add a write to file stage.
    # This stage writes all messages to a file.
    pipeline.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))

    # Run the pipeline.
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
