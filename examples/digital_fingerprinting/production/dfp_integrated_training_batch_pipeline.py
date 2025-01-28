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
from datetime import datetime

import click

import morpheus.loaders  # noqa: F401 # pylint:disable=unused-import
import morpheus.modules  # noqa: F401 # pylint:disable=unused-import
# When segment modules are imported, they're added to the module registry.
# To avoid flake8 warnings about unused code, the noqa flag is used during import
import morpheus_dfp.modules  # noqa: F401 # pylint:disable=unused-import
from morpheus.cli.utils import get_log_levels
from morpheus.cli.utils import parse_log_level
from morpheus.config import Config
from morpheus.pipeline.pipeline import Pipeline
from morpheus.stages.general.multi_port_modules_stage import MultiPortModulesStage
from morpheus.stages.input.control_message_file_source_stage import ControlMessageFileSourceStage
from morpheus_dfp.utils.config_generator import ConfigGenerator
from morpheus_dfp.utils.config_generator import generate_ae_config
from morpheus_dfp.utils.dfp_arg_parser import DFPArgParser
from morpheus_dfp.utils.schema_utils import Schema
from morpheus_dfp.utils.schema_utils import SchemaBuilder


@click.command()
@click.option(
    "--source",
    type=click.Choice(["duo", "azure"], case_sensitive=False),
    required=True,
    help=("Indicates what type of logs are going to be used in the workload."),
)
@click.option(
    "--train_users",
    type=click.Choice(["all", "generic", "individual"], case_sensitive=False),
    help=("Indicates whether or not to train per user or a generic model for all users. "
          "Selecting none runs the inference pipeline."),
)
@click.option(
    "--skip_user",
    multiple=True,
    type=str,
    help="User IDs to skip. Mutually exclusive with only_user",
)
@click.option(
    "--only_user",
    multiple=True,
    type=str,
    help="Only users specified by this option will be included. Mutually exclusive with skip_user",
)
@click.option(
    "--start_time",
    type=click.DateTime(
        formats=['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%d %H:%M:%S%z']),
    default=None,
    help="The start of the time window, if undefined start_date will be `now()-duration`",
)
@click.option(
    "--duration",
    type=str,
    default="60d",
    help="The training duration to run starting from start_time",
)
@click.option(
    "--cache_dir",
    type=str,
    default="./.cache/dfp",
    show_envvar=True,
    help="The location to cache data such as S3 downloads and pre-processed data",
)
@click.option("--log_level",
              default="INFO",
              type=click.Choice(get_log_levels(), case_sensitive=False),
              callback=parse_log_level,
              help="Specify the logging level to use.")
@click.option("--sample_rate_s",
              type=int,
              default=0,
              show_envvar=True,
              help="Minimum time step, in milliseconds, between object logs.")
@click.option('--tracking_uri',
              type=str,
              default="http://mlflow:5000",
              help=("The MLflow tracking URI to connect to the tracking backend."))
@click.option('--mlflow_experiment_name_template',
              type=str,
              default=None,
              help=("The MLflow experiment name template to use when logging experiments."
                    "If None, defaults to dfp/source/training/{reg_model_name}"))
@click.option('--mlflow_model_name_template',
              type=str,
              default=None,
              help=("The MLflow model name template to use when logging models."
                    "If None, defaults to DFP-source-{user_id}"))
@click.option("--disable_pre_filtering",
              is_flag=True,
              help=("Enabling this option will skip pre-filtering of json messages. "
                    "This is only useful when inputs are known to be valid json."))
@click.option(
    "--input_file",
    type=str,
    multiple=True,
    help=("List of control message defination files to process. Can specify multiple arguments for multiple files."
          "Also accepts glob (*) wildcards"
          "Refer to fsspec documentation for list of possible options."),
)
@click.option('--silence_monitors', flag_value=True, help='Controls whether monitors will be verbose.')
def run_pipeline(source: str,
                 train_users: str,
                 skip_user: tuple[str],
                 only_user: tuple[str],
                 start_time: datetime | None,
                 duration: str,
                 cache_dir: str,
                 log_level: int,
                 sample_rate_s: int,
                 tracking_uri: str,
                 silence_monitors: bool,
                 mlflow_experiment_name_template: str | None,
                 mlflow_model_name_template: str | None,
                 **kwargs):
    if (skip_user and only_user):
        logging.error("Option --skip_user and --only_user are mutually exclusive. Exiting")

    if mlflow_experiment_name_template is None:
        mlflow_experiment_name_template = f'dfp/{source}/training/' + '{reg_model_name}'
    if mlflow_model_name_template is None:
        mlflow_model_name_template = f'DFP-{source}-' + '{user_id}'
    dfp_arg_parser = DFPArgParser(skip_user,
                                  only_user,
                                  start_time,
                                  log_level,
                                  cache_dir,
                                  sample_rate_s,
                                  duration,
                                  source,
                                  tracking_uri,
                                  silence_monitors,
                                  mlflow_experiment_name_template,
                                  mlflow_model_name_template,
                                  train_users)

    dfp_arg_parser.init()

    # Default user_id column -- override with ControlMessage
    userid_column_name = "username"
    # Default timestamp column -- override with ControlMessage
    timestamp_column_name = "timestamp"

    config: Config = generate_ae_config(source, userid_column_name, timestamp_column_name)

    # Construct the data frame Schema used to normalize incoming data
    schema_builder = SchemaBuilder(config, source)
    schema: Schema = schema_builder.build_schema()

    # Create config helper used to generate config parameters for the DFP module
    # This will populate to the minimum configuration parameters with intelligent default values
    config_generator = ConfigGenerator(config, dfp_arg_parser, schema)

    dfp_deployment_module_config = config_generator.get_module_conf()

    #                                     +--------------------------------------+
    #                                     |  control_message_file_source_stage   |
    #                                     +--------------------------------------+
    #                                                        |
    #                                                        v
    #                                          +------------------------+
    #                                          |  dfp_deployment_module |
    # +-------------------------------------------------------------------------------------------------------------+
    # |                                                      |                                                      |
    # |                                                      v                                                      |
    # |                                   +-------------------------------------+                                   |
    # |                                   |       fsspec_loader_module          |                                   |
    # |                                   +-------------------------------------+                                   |
    # |                                                      |                                                      |
    # |                                                      v                                                      |
    # |                                   +-------------------------------------+                                   |
    # |                                   |              broadcast              |                                   |
    # |                                   +-------------------------------------+                                   |
    # |                                              /                \                                             |
    # |                                             /                  \                                            |
    # |                                            /                    \                                           |
    # |                                           v                      v                                          |
    # |                          +-------------------------+    +-------------------------+                         |
    # |                          |dfp_trianing_pipe_module |    |dfp_inference_pipe_module|                         |
    # |   +------------------------------------------------+    +----------------------------------------------+    |
    # |   |                                                |    |                                               |   |
    # |   |      +-------------------------------------+   |    |    + -------------------------------------+   |   |
    # |   |      |           preproc_module            |   |    |    |           preproc_module             |   |   |
    # |   |      +-------------------------------------+   |    |    + -------------------------------------+   |   |
    # |   |                        |                       |    |                        |                      |   |
    # |   |                        v                       |    |                        v                      |   |
    # |   |      +-------------------------------------+   |    |    + -------------------------------------+   |   |
    # |   |      |     dfp_rolling_window_module       |   |    |    |      dfp_rolling_window_module      |    |   |
    # |   |      +-------------------------------------+   |    |    + -------------------------------------+   |   |
    # |   |                        |                       |    |                        |                      |   |
    # |   |                        v                       |    |                        v                      |   |
    # |   |      +-------------------------------------+   |    |    + -------------------------------------+   |   |
    # |   |      |        dfp_data_prep_module         |   |    |    |         dfp_data_prep_module         |   |   |
    # |   |      +-------------------------------------+   |    |    + -------------------------------------+   |   |
    # |   |                        |                       |    |                        |                      |   |
    # |   |                        v                       |    |                        v                      |   |
    # |   |      +-------------------------------------+   |    |    + -------------------------------------+   |   |
    # |   |      |           monitor_module            |   |    |    |             monitor_module           |   |   |
    # |   |      +-------------------------------------+   |    |    + -------------------------------------+   |   |
    # |   |                        |                       |    |                        |                      |   |
    # |   |                        v                       |    |                        v                      |   |
    # |   |      +-------------------------------------+   |    |    + -------------------------------------+   |   |
    # |   |      |        dfp_training_module          |   |    |    |         dfp_inference_module         |   |   |
    # |   |      +-------------------------------------+   |    |    + -------------------------------------+   |   |
    # |   |                        |                       |    |                        |                      |   |
    # |   |                        v                       |    |                        v                      |   |
    # |   |      +-------------------------------------+   |    |    + -------------------------------------+   |   |
    # |   |      |           monitor_module            |   |    |    |             monitor_module          |    |   |
    # |   |      +-------------------------------------+   |    |    + -------------------------------------+   |   |
    # |   |                        |                       |    |                        |                      |   |
    # |   |                        v                       |    |                        v                      |   |
    # |   |      +-------------------------------------+   |    |    + -------------------------------------+   |   |
    # |   |      |       mlflow_model_writer_module    |   |    |    |        filter_detections_module      |   |   |
    # |   |      +-------------------------------------+   |    |    + -------------------------------------+   |   |
    # |   |                        |                       |    |                        |                      |   |
    # |   |                        v                       |    |                        v                      |   |
    # |   |      +-------------------------------------+   |    |    + -------------------------------------+   |   |
    # |   |      |            monitor_module           |   |    |    |        dfp_post_proc_module          |   |   |
    # |   |      +-------------------------------------+   |    |    + -------------------------------------+   |   |
    # |    ------------------------------------------------     |                        |                      |   |
    # |                                                         |                        v                      |   |
    # |                                                         |    +-------------------------------------+    |   |
    # |                                                         |    |          serialize_module           |    |   |
    # |                                                         |    +-------------------------------------+    |   |
    # |                                                         |                        |                      |   |
    # |                                                         |                        v                      |   |
    # |                                                         |    +-------------------------------------+    |   |
    # |                                                         |    |      write_to_file_module           |    |   |
    # |                                                         |    +-------------------------------------+    |   |
    # |                                                         |                        |                      |   |
    # |                                                         |                        v                      |   |
    # |                                                         |    +-------------------------------------+    |   |
    # |                                                         |    |            monitor_module           |    |   |
    # |                                                         |    +-------------------------------------+    |   |
    # |                                                         ------------------------------------------------    |
    #  --------------------------------------------------------------------------------------------------------------

    # Create a pipeline object
    pipeline = Pipeline(config)

    source_stage = pipeline.add_stage(ControlMessageFileSourceStage(config, filenames=list(kwargs["input_file"])))

    dfp_deployment_stage = pipeline.add_stage(
        MultiPortModulesStage(config,
                              dfp_deployment_module_config,
                              input_ports=["input"],
                              output_ports=["output_0", "output_1"]))

    pipeline.add_edge(source_stage, dfp_deployment_stage)

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline(obj={}, auto_envvar_prefix='DFP', show_default=True, prog_name="dfp")
