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
import yaml
from stages.create_features import CreateFeaturesRWStage
from stages.preprocessing import PreprocessingRWStage

from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import MessageMeta
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.pipeline.stage_decorator import stage
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.input.appshield_source_stage import AppShieldSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.add_scores_stage import AddScoresStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.utils.logger import configure_logging

CUR_DIR = os.path.dirname(__file__)


@click.command()
@click.option('--debug', default=False)
@click.option(
    "--num_threads",
    default=len(os.sched_getaffinity(0)),
    type=click.IntRange(min=1),
    help="Number of internal pipeline threads to use.",
)
@click.option(
    "--n_dask_workers",
    default=6,
    type=click.IntRange(min=1),
    help="Number of dask workers.",
)
@click.option(
    "--threads_per_dask_worker",
    default=2,
    type=click.IntRange(min=1),
    help="Number of threads per each dask worker.",
)
@click.option(
    "--model_max_batch_size",
    default=1024,
    type=click.IntRange(min=1),
    help="Max batch size to use for the model.",
)
@click.option(
    "--pipeline_batch_size",
    default=1024,
    type=click.IntRange(min=1),
    help=("Internal batch size for the pipeline. Can be much larger than the model batch size."),
)
@click.option(
    "--conf_file",
    type=click.STRING,
    default=os.path.join(CUR_DIR, "config/ransomware_detection.yaml"),
    help="Ransomware detection configuration filepath.",
)
@click.option(
    "--model_name",
    default="ransomw-model-short-rf",
    help="The name of the model that is deployed on Tritonserver.",
)
@click.option("--server_url", required=True, help="Tritonserver url.")
@click.option(
    "--sliding_window",
    default=3,
    type=click.IntRange(min=3),
    help="Sliding window to be used for model input request.",
)
@click.option(
    '--input_glob',
    type=str,
    required=True,
    help=("Input glob pattern to match files to read. For example, './input_dir/*/snapshot-*/*.json' would read all "
          "files with the 'json' extension in the directory 'input_dir'."))
@click.option('--watch_directory',
              type=bool,
              default=False,
              help=("The watch directory option instructs this stage to not close down once all files have been read. "
                    "Instead it will read all files that match the 'input_glob' pattern, and then continue to watch "
                    "the directory for additional files. Any new files that are added that match the glob will then "
                    "be processed."))
@click.option(
    "--output_file",
    type=click.STRING,
    default=".tmp/output/ransomware_detection_output.jsonlines",
    help="The path to the file where the inference output will be saved.",
)
def run_pipeline(debug: bool,
                 num_threads: int,
                 n_dask_workers: int,
                 threads_per_dask_worker: int,
                 model_max_batch_size: int,
                 pipeline_batch_size: int,
                 conf_file: str,
                 model_name: str,
                 server_url: str,
                 sliding_window: int,
                 input_glob: str,
                 watch_directory: bool,
                 output_file: str):

    if debug:
        configure_logging(log_level=logging.DEBUG)
    else:
        configure_logging(log_level=logging.INFO)

    snapshot_fea_length = 99

    # Its necessary to get the global config object and configure it for FIL mode.
    config = Config()
    config.mode = PipelineModes.FIL

    # Below properties are specified by the command line.
    config.num_threads = num_threads
    config.model_max_batch_size = model_max_batch_size
    config.pipeline_batch_size = pipeline_batch_size
    config.feature_length = snapshot_fea_length * sliding_window
    config.class_labels = ["pred", "score"]

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    # Load ransomware detection configuration.
    rwd_conf = load_yaml(conf_file)

    # Exclude columns that are not required.
    cols_exclude = ["SHA256"]

    # Only intrested plugins files will be read from Appshield snapshots.
    interested_plugins = ['ldrmodules', 'threadlist', 'envars', 'vadinfo', 'handles']

    # Columns from the above intrested plugins.
    cols_interested_plugins = rwd_conf['raw_columns']

    # Feature columns used by the model.
    model_features = rwd_conf['model_features']

    # Features to include in the DF, superset of model_features along with a few that the model doesn't receive
    feature_columns = model_features + rwd_conf['features']

    # File extensions.
    file_extns = rwd_conf['file_extensions']

    # Set source stage.
    # This stage reads raw data from the required plugins and merge all the plugins data into a single dataframe
    # for a given source.
    pipeline.set_source(
        AppShieldSourceStage(
            config,
            input_glob,
            interested_plugins,
            cols_interested_plugins,
            cols_exclude=cols_exclude,
            watch_directory=watch_directory,
        ))

    # Add a monitor stage.
    # This stage logs the metrics (msg/sec) from the above stage.
    pipeline.add_stage(MonitorStage(config, description="FromFile rate"))

    # Add a create features stage.
    # This stage generates model feature values from the raw data.
    pipeline.add_stage(
        CreateFeaturesRWStage(config,
                              interested_plugins,
                              feature_columns,
                              file_extns,
                              n_workers=n_dask_workers,
                              threads_per_worker=threads_per_dask_worker))

    # Add a monitor stage.
    # This stage logs the metrics (msg/sec) from the above stage.
    pipeline.add_stage(MonitorStage(config, description="CreateFeatures rate"))

    # Add preprocessing stage.
    # This stage generates snapshot sequences using sliding window for each pid_process.
    pipeline.add_stage(PreprocessingRWStage(config, feature_columns=model_features, sliding_window=sliding_window))

    # Add a monitor stage
    # This stage logs the metrics (msg/sec) from the above stage.
    pipeline.add_stage(MonitorStage(config, description="PreProcessing rate"))

    # Add a inference stage.
    # This stage sends inference requests to the Tritonserver and captures the response.
    pipeline.add_stage(
        TritonInferenceStage(
            config,
            model_name=model_name,
            server_url=server_url,
            force_convert_inputs=True,
            thread_count=1  # Work-around for issue #1891 remove once resolved.
        ))

    # Add a monitor stage.
    # This stage logs the metrics (msg/sec) from the above stage.
    pipeline.add_stage(MonitorStage(config, description="Inference rate"))

    # Add a add scores stage.
    # This stage adds probability scores to each message.
    pipeline.add_stage(AddScoresStage(config, labels=["score"]))

    # Add a monitor stage.
    # This stage logs the metrics (msg/sec) from the above stage.
    pipeline.add_stage(MonitorStage(config, description="AddScore rate"))

    # Add a serialize stage.
    # This stage includes & excludes columns from messages.
    pipeline.add_stage(SerializeStage(config, exclude=[r'^ID$', r'^_ts_', r'source_pid_process']))

    # Add a monitor stage.
    # This stage logs the metrics (msg/sec) from the above stage.
    pipeline.add_stage(MonitorStage(config, description="Serialize rate"))

    @stage(needed_columns={'timestamp_process': TypeId.STRING})
    def concat_columns(msg: MessageMeta) -> MessageMeta:
        """
        This stage concatinates the timestamp and pid_process columns to create a unique field.
        """
        with msg.mutable_dataframe() as df:
            df['timestamp_process'] = df['timestamp'] + df['pid_process']

        return msg

    pipeline.add_stage(concat_columns(config))

    # Add a write file stage.
    # This stage writes all messages to a file.
    pipeline.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))

    # Add a monitor stage.
    # This stage logs the metrics (msg/sec) from the above stage.
    pipeline.add_stage(MonitorStage(config, description="ToFile rate"))

    # Run the pipeline.
    pipeline.run()


def load_yaml(filepath: str) -> typing.Dict[object, object]:
    """
    This function loads yaml configuration to a dictionary

    Parameters
        ----------
        filepath : str
            A file's path

        Returns
        -------
        typing.Dict[object, object]
            Configuration as a dictionary
    """
    with open(filepath, 'r', encoding='utf8') as f:
        conf_dct = yaml.safe_load(f)
        f.close()

    return conf_dct


# Execution starts here
if __name__ == "__main__":
    run_pipeline()
