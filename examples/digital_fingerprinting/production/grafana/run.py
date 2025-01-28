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
"""DFP training & inference pipelines for Azure Active Directory logs."""

import functools
import logging
import logging.handlers
import os
import typing
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import click
import logging_loki
import mlflow
import pandas as pd

from morpheus.cli.utils import get_log_levels
from morpheus.cli.utils import get_package_relative_file
from morpheus.cli.utils import parse_log_level
from morpheus.common import FileTypes
from morpheus.common import FilterSource
from morpheus.config import Config
from morpheus.config import ConfigAutoEncoder
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.filter_detections_stage import FilterDetectionsStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.utils.column_info import ColumnInfo
from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.column_info import DateTimeColumn
from morpheus.utils.column_info import DistinctIncrementColumn
from morpheus.utils.column_info import IncrementColumn
from morpheus.utils.column_info import RenameColumn
from morpheus.utils.column_info import StringCatColumn
from morpheus.utils.file_utils import date_extractor
from morpheus.utils.file_utils import load_labels_file
from morpheus.utils.logger import configure_logging
from morpheus_dfp.stages.dfp_file_batcher_stage import DFPFileBatcherStage
from morpheus_dfp.stages.dfp_file_to_df import DFPFileToDataFrameStage
from morpheus_dfp.stages.dfp_inference_stage import DFPInferenceStage
from morpheus_dfp.stages.dfp_mlflow_model_writer import DFPMLFlowModelWriterStage
from morpheus_dfp.stages.dfp_postprocessing_stage import DFPPostprocessingStage
from morpheus_dfp.stages.dfp_preprocessing_stage import DFPPreprocessingStage
from morpheus_dfp.stages.dfp_rolling_window_stage import DFPRollingWindowStage
from morpheus_dfp.stages.dfp_split_users_stage import DFPSplitUsersStage
from morpheus_dfp.stages.dfp_training import DFPTraining
from morpheus_dfp.stages.multi_file_source import MultiFileSource
from morpheus_dfp.utils.regex_utils import iso_date_regex


def _file_type_name_to_enum(file_type: str) -> FileTypes:
    """Converts a file type name to a FileTypes enum."""
    if (file_type == "JSON"):
        return FileTypes.JSON
    if (file_type == "CSV"):
        return FileTypes.CSV
    if (file_type == "PARQUET"):
        return FileTypes.PARQUET

    return FileTypes.Auto


@click.command()
@click.option(
    "--train_users",
    type=click.Choice(["all", "generic", "individual", "none"], case_sensitive=False),
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
    help="The duration to run starting from start_time",
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
@click.option("--filter_threshold",
              type=float,
              default=2.0,
              show_envvar=True,
              help="Filter out inference results below this threshold")
@click.option(
    "--input_file",
    "-f",
    type=str,
    multiple=True,
    help=("List of files to process. Can specify multiple arguments for multiple files. "
          "Also accepts glob (*) wildcards and schema prefixes such as `s3://`. "
          "For example, to make a local cache of an s3 bucket, use `filecache::s3://mybucket/*`. "
          "Refer to fsspec documentation for list of possible options."),
)
@click.option("--file_type_override",
              "-t",
              type=click.Choice(["AUTO", "JSON", "CSV", "PARQUET"], case_sensitive=False),
              default="JSON",
              help="Override the detected file type. Values can be 'AUTO', 'JSON', 'CSV', or 'PARQUET'.",
              callback=lambda _, __, value: None if value is None else _file_type_name_to_enum(value))
@click.option('--watch_inputs',
              type=bool,
              is_flag=True,
              default=False,
              help=("Instructs the pipeline to continuously check the paths specified by `--input_file` for new files. "
                    "This assumes that the at least one paths contains a wildcard."))
@click.option("--watch_interval",
              type=float,
              default=1.0,
              help=("Amount of time, in seconds, to wait between checks for new files. "
                    "Only used if --watch_inputs is set."))
@click.option('--tracking_uri',
              type=str,
              default="http://mlflow:5000",
              help=("The MLflow tracking URI to connect to the tracking backend."))
@click.option('--mlflow_experiment_name_template',
              type=str,
              default="dfp/azure/training/{reg_model_name}",
              help="The MLflow experiment name template to use when logging experiments. ")
@click.option('--mlflow_model_name_template',
              type=str,
              default="DFP-azure-{user_id}",
              help="The MLflow model name template to use when logging models. ")
@click.option('--use_postproc_schema', is_flag=True, help='Assume that input data has already been preprocessed.')
@click.option('--inference_detection_file_name', type=str, default="dfp_detections_azure.csv")
@click.option('--loki_url',
              type=str,
              default="http://loki:3100",
              help=("Loki URL for error logging and alerting in Grafana."))
def run_pipeline(train_users,
                 skip_user: typing.Tuple[str],
                 only_user: typing.Tuple[str],
                 start_time: datetime,
                 duration,
                 cache_dir,
                 log_level,
                 sample_rate_s,
                 filter_threshold,
                 mlflow_experiment_name_template,
                 mlflow_model_name_template,
                 file_type_override,
                 use_postproc_schema,
                 inference_detection_file_name,
                 loki_url,
                 **kwargs):
    """Runs the DFP pipeline."""
    # To include the generic, we must be training all or generic
    include_generic = train_users in ("all", "generic")

    # To include individual, we must be either training or inferring
    include_individual = train_users != "generic"

    # None indicates we aren't training anything
    is_training = train_users != "none"

    skip_users = list(skip_user)
    only_users = list(only_user)

    duration = timedelta(seconds=pd.Timedelta(duration).total_seconds())
    if start_time is None:
        end_time = datetime.now(tz=timezone.utc)
        start_time = end_time - duration
    else:
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)

        end_time = start_time + duration

    # Enable the Morpheus logger
    loki_handler = logging_loki.LokiHandler(
        url=f"{loki_url}/loki/api/v1/push",
        tags={"app": "morpheus"},
        version="1",
    )
    configure_logging(loki_handler, log_level=log_level)
    logging.getLogger("mlflow").setLevel(log_level)

    if (len(skip_users) > 0 and len(only_users) > 0):
        logging.error("Option --skip_user and --only_user are mutually exclusive. Exiting")

    logger = logging.getLogger(f"morpheus.{__name__}")

    logger.info("Running training pipeline with the following options: ")
    logger.info("Train generic_user: %s", include_generic)
    logger.info("Skipping users: %s", skip_users)
    logger.info("Start Time: %s", start_time)
    logger.info("Duration: %s", duration)
    logger.info("Cache Dir: %s", cache_dir)

    if ("tracking_uri" in kwargs):
        # Initialize ML Flow
        mlflow.set_tracking_uri(kwargs["tracking_uri"])
        logger.info("Tracking URI: %s", mlflow.get_tracking_uri())

    config = Config()
    config.num_threads = len(os.sched_getaffinity(0))

    config.ae = ConfigAutoEncoder()

    config.ae.feature_columns = load_labels_file(get_package_relative_file("data/columns_ae_azure.txt"))
    config.ae.userid_column_name = "username"
    config.ae.timestamp_column_name = "timestamp"

    # Specify the column names to ensure all data is uniform
    if (use_postproc_schema):

        source_column_info = [
            ColumnInfo(name="autonomousSystemNumber", dtype=str),
            ColumnInfo(name="location_geoCoordinates_latitude", dtype=float),
            ColumnInfo(name="location_geoCoordinates_longitude", dtype=float),
            ColumnInfo(name="resourceDisplayName", dtype=str),
            ColumnInfo(name="travel_speed_kmph", dtype=float),
            DateTimeColumn(name=config.ae.timestamp_column_name, dtype=datetime, input_name="time"),
            ColumnInfo(name="appDisplayName", dtype=str),
            ColumnInfo(name="clientAppUsed", dtype=str),
            RenameColumn(name=config.ae.userid_column_name, dtype=str, input_name="userPrincipalName"),
            RenameColumn(name="deviceDetailbrowser", dtype=str, input_name="deviceDetail_browser"),
            RenameColumn(name="deviceDetaildisplayName", dtype=str, input_name="deviceDetail_displayName"),
            RenameColumn(name="deviceDetailoperatingSystem", dtype=str, input_name="deviceDetail_operatingSystem"),

            # RenameColumn(name="location_country", dtype=str, input_name="location_countryOrRegion"),
            ColumnInfo(name="location_city_state_country", dtype=str),
            ColumnInfo(name="location_state_country", dtype=str),
            ColumnInfo(name="location_country", dtype=str),

            # Non-features
            ColumnInfo(name="is_corp_vpn", dtype=bool),
            ColumnInfo(name="distance_km", dtype=float),
            ColumnInfo(name="ts_delta_hour", dtype=float),
        ]
        source_schema = DataFrameInputSchema(column_info=source_column_info)

        preprocess_column_info = [
            ColumnInfo(name=config.ae.timestamp_column_name, dtype=datetime),
            ColumnInfo(name=config.ae.userid_column_name, dtype=str),

            # Resource access
            ColumnInfo(name="appDisplayName", dtype=str),
            ColumnInfo(name="resourceDisplayName", dtype=str),
            ColumnInfo(name="clientAppUsed", dtype=str),

            # Device detail
            ColumnInfo(name="deviceDetailbrowser", dtype=str),
            ColumnInfo(name="deviceDetaildisplayName", dtype=str),
            ColumnInfo(name="deviceDetailoperatingSystem", dtype=str),

            # Location information
            ColumnInfo(name="autonomousSystemNumber", dtype=str),
            ColumnInfo(name="location_geoCoordinates_latitude", dtype=float),
            ColumnInfo(name="location_geoCoordinates_longitude", dtype=float),
            ColumnInfo(name="location_city_state_country", dtype=str),
            ColumnInfo(name="location_state_country", dtype=str),
            ColumnInfo(name="location_country", dtype=str),

            # Derived information
            ColumnInfo(name="travel_speed_kmph", dtype=float),

            # Non-features
            ColumnInfo(name="is_corp_vpn", dtype=bool),
            ColumnInfo(name="distance_km", dtype=float),
            ColumnInfo(name="ts_delta_hour", dtype=float),
        ]

        preprocess_schema = DataFrameInputSchema(column_info=preprocess_column_info, preserve_columns=["_batch_id"])

        exclude_from_training = [
            config.ae.userid_column_name,
            config.ae.timestamp_column_name,
            "is_corp_vpn",
            "distance_km",
            "ts_delta_hour",
        ]

        config.ae.feature_columns = [
            name for (name, dtype) in preprocess_schema.output_columns if name not in exclude_from_training
        ]
    else:
        source_column_info = [
            DateTimeColumn(name=config.ae.timestamp_column_name, dtype=datetime, input_name="time"),
            RenameColumn(name=config.ae.userid_column_name, dtype=str, input_name="properties.userPrincipalName"),
            RenameColumn(name="appDisplayName", dtype=str, input_name="properties.appDisplayName"),
            ColumnInfo(name="category", dtype=str),
            RenameColumn(name="clientAppUsed", dtype=str, input_name="properties.clientAppUsed"),
            RenameColumn(name="deviceDetailbrowser", dtype=str, input_name="properties.deviceDetail.browser"),
            RenameColumn(name="deviceDetaildisplayName", dtype=str, input_name="properties.deviceDetail.displayName"),
            RenameColumn(name="deviceDetailoperatingSystem",
                         dtype=str,
                         input_name="properties.deviceDetail.operatingSystem"),
            StringCatColumn(name="location",
                            dtype=str,
                            input_columns=[
                                "properties.location.city",
                                "properties.location.countryOrRegion",
                            ],
                            sep=", "),
            RenameColumn(name="statusfailureReason", dtype=str, input_name="properties.status.failureReason"),
        ]

        source_schema = DataFrameInputSchema(json_columns=["properties"], column_info=source_column_info)

        # Preprocessing schema
        preprocess_column_info = [
            ColumnInfo(name=config.ae.timestamp_column_name, dtype=datetime),
            ColumnInfo(name=config.ae.userid_column_name, dtype=str),
            ColumnInfo(name="appDisplayName", dtype=str),
            ColumnInfo(name="clientAppUsed", dtype=str),
            ColumnInfo(name="deviceDetailbrowser", dtype=str),
            ColumnInfo(name="deviceDetaildisplayName", dtype=str),
            ColumnInfo(name="deviceDetailoperatingSystem", dtype=str),
            ColumnInfo(name="statusfailureReason", dtype=str),

            # Derived columns
            IncrementColumn(name="logcount",
                            dtype=int,
                            input_name=config.ae.timestamp_column_name,
                            groupby_column=config.ae.userid_column_name),
            DistinctIncrementColumn(name="locincrement",
                                    dtype=int,
                                    input_name="location",
                                    groupby_column=config.ae.userid_column_name,
                                    timestamp_column=config.ae.timestamp_column_name),
            DistinctIncrementColumn(name="appincrement",
                                    dtype=int,
                                    input_name="appDisplayName",
                                    groupby_column=config.ae.userid_column_name,
                                    timestamp_column=config.ae.timestamp_column_name)
        ]

        preprocess_schema = DataFrameInputSchema(column_info=preprocess_column_info, preserve_columns=["_batch_id"])

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    pipeline.set_source(
        MultiFileSource(config,
                        filenames=list(kwargs["input_file"]),
                        watch=kwargs["watch_inputs"],
                        watch_interval=kwargs["watch_interval"]))

    # Batch files into batches by time. Use the default ISO date extractor from the filename
    pipeline.add_stage(
        DFPFileBatcherStage(config,
                            period="D",
                            sampling_rate_s=sample_rate_s,
                            date_conversion_func=functools.partial(date_extractor, filename_regex=iso_date_regex),
                            start_time=start_time,
                            end_time=end_time))

    parser_kwargs = None
    if (file_type_override == FileTypes.JSON):
        parser_kwargs = {"lines": False, "orient": "records"}
    # Output is a list of fsspec files. Convert to DataFrames. This caches downloaded data
    pipeline.add_stage(
        DFPFileToDataFrameStage(
            config,
            schema=source_schema,
            file_type=file_type_override,
            parser_kwargs=parser_kwargs,  # TODO(Devin) probably should be configurable too
            cache_dir=cache_dir))

    pipeline.add_stage(MonitorStage(config, description="Input data rate"))

    # This will split users or just use one single user
    pipeline.add_stage(
        DFPSplitUsersStage(config,
                           include_generic=include_generic,
                           include_individual=include_individual,
                           skip_users=skip_users,
                           only_users=only_users))

    # Next, have a stage that will create rolling windows
    pipeline.add_stage(
        DFPRollingWindowStage(
            config,
            min_history=300 if is_training else 1,
            min_increment=300 if is_training else 0,
            # For inference, we only ever want 1 day max
            max_history="60d" if is_training else "1d",
            cache_dir=cache_dir))

    # Output is UserMessageMeta -- Cached frame set
    pipeline.add_stage(DFPPreprocessingStage(config, input_schema=preprocess_schema))

    model_name_formatter = mlflow_model_name_template
    experiment_name_formatter = mlflow_experiment_name_template

    if (is_training):
        # Finally, perform training which will output a model
        pipeline.add_stage(DFPTraining(config, epochs=100, validation_size=0.15))

        pipeline.add_stage(MonitorStage(config, description="Training rate", smoothing=0.001))

        # Write that model to MLFlow
        pipeline.add_stage(
            DFPMLFlowModelWriterStage(config,
                                      model_name_formatter=model_name_formatter,
                                      experiment_name_formatter=experiment_name_formatter))
    else:
        # Perform inference on the preprocessed data
        pipeline.add_stage(DFPInferenceStage(config, model_name_formatter=model_name_formatter))

        pipeline.add_stage(MonitorStage(config, description="Inference rate", smoothing=0.001))

        # Filter for only the anomalous logs
        pipeline.add_stage(
            FilterDetectionsStage(config,
                                  threshold=filter_threshold,
                                  filter_source=FilterSource.DATAFRAME,
                                  field_name='mean_abs_z'))
        pipeline.add_stage(DFPPostprocessingStage(config))

        # Exclude the columns we don't want in our output
        pipeline.add_stage(SerializeStage(config, exclude=['batch_count', 'origin_hash', '_row_hash', '_batch_id']))

        # Write all anomalies to a CSV file
        pipeline.add_stage(WriteToFileStage(config, filename=inference_detection_file_name, overwrite=True))

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    run_pipeline(obj={}, auto_envvar_prefix='DFP', show_default=True, prog_name="dfp")
