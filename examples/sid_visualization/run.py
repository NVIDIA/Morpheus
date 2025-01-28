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
import typing

import click
import mrc

from morpheus.common import FileTypes
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages import MessageMeta
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.postprocess.add_classifications_stage import AddClassificationsStage
from morpheus.stages.postprocess.generate_viz_frames_stage import GenerateVizFramesStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage
from morpheus.utils.file_utils import get_data_file_path
from morpheus.utils.file_utils import load_labels_file
from morpheus.utils.logger import configure_logging


class NLPVizFileSource(PreallocatorMixin, SingleOutputSource):
    """
    Source stage is used to load messages from a file and dumping the contents into the pipeline immediately. Useful for
    testing performance and accuracy of a pipeline.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    filename : str
        Name of the file from which the messages will be read.
    iterative: boolean
        Iterative mode will emit dataframes one at a time. Otherwise a list of dataframes is emitted. Iterative mode is
        good for interleaving source stages.
    file_type : `morpheus.common.FileTypes`, default = 'auto'
        Indicates what type of file to read. Specifying 'auto' will determine the file type from the extension.
        Supported extensions: 'json', 'csv'
    repeat: int, default = 1
        Repeats the input dataset multiple times. Useful to extend small datasets for debugging.
    filter_null: bool, default = True
        Whether or not to filter rows with null 'data' column. Null values in the 'data' column can cause issues down
        the line with processing. Setting this to True is recommended.
    """

    def __init__(self, c: Config, filenames: typing.List[str], file_type: FileTypes = FileTypes.Auto):

        super().__init__(c)

        self._batch_size = c.pipeline_batch_size

        self._filenames = filenames
        self._file_type = file_type

        self._input_count = None
        self._max_concurrent = c.num_threads

    @property
    def name(self) -> str:
        return "from-multi-file"

    @property
    def input_count(self) -> int:
        """Return None for no max intput count"""
        return self._input_count

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(MessageMeta)

    def supports_cpp_node(self):
        return False

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:

        if self._build_cpp_node():
            raise RuntimeError("Does not support C++ nodes")

        return builder.make_source(self.unique_name, self._generate_frames())

    def _generate_frames(self):

        for f in self._filenames:

            # Read the dataframe into memory
            df = read_file_to_df(
                f,
                self._file_type,
                filter_nulls=True,
                df_type="cudf",
            )

            # Truncate it down to the max size
            df = df.head(self._batch_size)

            x = MessageMeta(df)

            yield x


@click.command()
@click.option("--debug/--no-debug", default=False)
@click.option(
    "--num_threads",
    default=len(os.sched_getaffinity(0)),
    type=click.IntRange(min=1),
    help="Number of internal pipeline threads to use.",
)
@click.option(
    "--input_file",
    "-f",
    type=click.Path(exists=True, dir_okay=False),
    multiple=True,
    required=True,
    default=[
        "examples/data/sid_visualization/group1-benign-2nodes-v2.jsonlines",
        "examples/data/sid_visualization/group2-benign-50nodes.jsonlines"
    ],
    help="List of files to send to the visualization, in order.",
)
@click.option('--max_batch_size',
              default=50000,
              type=click.IntRange(min=1),
              help=("For each input_file, truncate the number of rows to this size."))
@click.option(
    "--model_name",
    default="sid-minibert-onnx",
    help="The name of the model that is deployed on Tritonserver.",
)
@click.option("--triton_server_url", default="localhost:8000", required=True, help="Tritonserver url.")
def run_pipeline(debug, num_threads, input_file, max_batch_size, model_name, triton_server_url):

    if debug:
        configure_logging(log_level=logging.DEBUG)
    else:
        configure_logging(log_level=logging.INFO)

    # Its necessary to get the global config object and configure it for FIL mode.
    config = Config()
    config.mode = PipelineModes.NLP

    # Below properties are specified by the command line.
    config.num_threads = num_threads
    config.pipeline_batch_size = max_batch_size
    config.feature_length = 256
    config.model_max_batch_size = 32
    config.class_labels = load_labels_file(get_data_file_path("data/labels_nlp.txt"))

    # Create a linear pipeline object.
    pipeline = LinearPipeline(config)

    # Set source stage.
    # This stage reads raw data from the required plugins and merge all the plugins data into a single dataframe
    # for a given source.
    pipeline.set_source(NLPVizFileSource(config, filenames=input_file))

    # Add a deserialize stage.
    # At this stage, messages were logically partitioned based on the 'pipeline_batch_size'.
    pipeline.add_stage(DeserializeStage(config))

    # Add a preprocessing NLP stage.
    # This stage preprocess the rows in the Dataframe.
    pipeline.add_stage(
        PreprocessNLPStage(config,
                           vocab_hash_file=get_data_file_path("data/bert-base-uncased-hash.txt"),
                           truncation=True,
                           do_lower_case=True,
                           add_special_tokens=False))

    # Add a inference stage.
    # This stage sends inference requests to the Tritonserver and captures the response.
    pipeline.add_stage(
        TritonInferenceStage(
            config,
            model_name=model_name,
            server_url=triton_server_url,
            force_convert_inputs=True,
        ))

    # Add a monitor stage.
    # This stage logs the metrics (msg/sec) from the above stage.
    pipeline.add_stage(MonitorStage(config, description="Inference rate"))

    # Add a add classification stage.
    # This stage adds detected classifications to each message.
    pipeline.add_stage(AddClassificationsStage(config, threshold=0.8))

    # Add a generate viz frame stage.
    # This stage writes out visualization DataFrames.
    pipeline.add_stage(GenerateVizFramesStage(config, server_url="0.0.0.0", server_port=8765))

    # Run the pipeline.
    pipeline.run()


# Execution starts here
if __name__ == "__main__":
    run_pipeline()
