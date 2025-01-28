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
from abp_pcap_preprocessing import AbpPcapPreprocessingStage

from morpheus.cli.commands import FILE_TYPE_NAMES
from morpheus.cli.utils import str_to_file_type
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.add_classifications_stage import AddClassificationsStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.logger import configure_logging

CUR_DIR = os.path.dirname(__file__)
EX_DATA_DIR = os.path.join(CUR_DIR, "../data")


@click.command()
@click.option(
    "--num_threads",
    default=len(os.sched_getaffinity(0)),
    type=click.IntRange(min=1),
    help="Number of internal pipeline threads to use.",
)
@click.option(
    "--pipeline_batch_size",
    default=100000,
    type=click.IntRange(min=1),
    help=("Internal batch size for the pipeline. Can be much larger than the model batch size. "
          "Also used for Kafka consumers."),
)
@click.option(
    "--model_max_batch_size",
    default=100000,
    type=click.IntRange(min=1),
    help="Max batch size to use for the model.",
)
@click.option(
    "--input_file",
    type=click.Path(exists=True, readable=True),
    default=os.path.join(EX_DATA_DIR, "abp_pcap_dump.jsonlines"),
    required=True,
    help="Input filepath.",
)
@click.option(
    "--output_file",
    default="./.tmp/output/pcap_out.jsonlines",
    help="The path to the file where the inference output will be saved.",
)
@click.option(
    "--model_fea_length",
    default=13,
    type=click.IntRange(min=1),
    help="Features length to use for the model.",
)
@click.option(
    "--model_name",
    default="abp-pcap-xgb",
    help="The name of the model that is deployed on Tritonserver.",
)
@click.option(
    "--iterative",
    is_flag=True,
    default=False,
    help=("Iterative mode will emit DataFrames one at a time. Otherwise a list of DataFrames is emitted. "
          "Iterative mode is good for interleaving source stages."),
)
@click.option("--server_url", required=True, help="Tritonserver url.", default="localhost:8000")
@click.option(
    "--file_type",
    type=click.Choice(FILE_TYPE_NAMES, case_sensitive=False),
    default="auto",
    help=("Indicates what type of file to read. "
          "Specifying 'auto' will determine the file type from the extension."),
)
def run_pipeline(
    num_threads,
    pipeline_batch_size,
    model_max_batch_size,
    model_fea_length,
    input_file,
    output_file,
    model_name,
    iterative,
    server_url,
    file_type,
):

    # Enable the default logger.
    configure_logging(log_level=logging.INFO)

    # Its necessary to get the global config object and configure it for FIL mode.
    config = Config()
    config.mode = PipelineModes.FIL

    # Below properties are specified by the command line.
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.feature_length = model_fea_length
    config.class_labels = ["probs"]

    # Create a linear pipeline object.
    pipeline = LinearPipeline(config)

    # Set source stage.
    # In this stage, messages were loaded from a file.
    pipeline.set_source(
        FileSourceStage(
            config,
            filename=input_file,
            iterative=iterative,
            file_type=str_to_file_type(file_type.lower()),
            filter_null=False,
        ))

    # Add a deserialize stage.
    # At this stage, messages were logically partitioned based on the 'pipeline_batch_size'.
    pipeline.add_stage(DeserializeStage(config))

    # Add the custom preprocessing stage.
    # This stage preprocess the rows in the Dataframe.
    pipeline.add_stage(AbpPcapPreprocessingStage(config))

    # Add a monitor stage.
    # This stage logs the metrics (msg/sec) from the above stage.
    pipeline.add_stage(MonitorStage(config, description="Preprocessing rate"))

    # Add a inference stage.
    # This stage sends inference requests to the Tritonserver and captures the response.
    pipeline.add_stage(TritonInferenceStage(config, model_name=model_name, server_url=server_url))

    # Add a monitor stage.
    # This stage logs the metrics (inf/sec) from the above stage.
    pipeline.add_stage(MonitorStage(config, description="Inference rate", unit="inf"))

    # Add a add classification stage.
    # This stage adds detected classifications to each message.
    pipeline.add_stage(AddClassificationsStage(config, labels=["probs"]))

    # Add a monitor stage.
    # This stage logs the metrics (msg/sec) from the above stage.
    pipeline.add_stage(MonitorStage(config, description="Add classification rate", unit="add-class"))

    # Add a serialize stage.
    # This stage includes & excludes columns from messages.
    pipeline.add_stage(SerializeStage(config))

    # Add a monitor stage.
    # This stage logs the metrics (msg/sec) from the above stage.
    pipeline.add_stage(MonitorStage(config, description="Serialize rate", unit="ser"))

    # Add a write to file stage.
    # This stage writes all messages to a file.
    pipeline.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))

    # Add a monitor stage.
    # This stage logs the metrics (msg/sec) from the above stage.
    pipeline.add_stage(MonitorStage(config, description="Write to file rate", unit="to-file"))

    # Run the pipeline.
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
