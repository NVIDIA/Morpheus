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

import datetime
import logging
import re
import typing
import warnings

import fsspec
import fsspec.utils
import mrc
import pandas as pd
from mrc.core import operators as ops

from morpheus.messages import ControlMessage
from morpheus.utils.file_utils import date_extractor
from morpheus.utils.loader_ids import FILE_TO_DF_LOADER
from morpheus.utils.module_ids import FILE_BATCHER
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_utils import merge_dictionaries
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)

DEFAULT_ISO_DATE_REGEX_PATTERN = (
    r"(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})"
    r"T(?P<hour>\d{1,2})(:|_)(?P<minute>\d{1,2})(:|_)(?P<second>\d{1,2})(?P<microsecond>\.\d{1,6})?Z")


@register_module(FILE_BATCHER, MORPHEUS_MODULE_NAMESPACE)
def file_batcher(builder: mrc.Builder):
    """
    This module loads the input files, removes files that are older than the chosen window of time,
    and then groups the remaining files by period that fall inside the window.


    Parameters
    ----------
    builder: mrc.Builder
        An mrc Builder object.

    Notes
    -----
        Configurable Parameters:
            - batching_options (dict): Options for batching; See below; Default: -
            - cache_dir (str): Cache directory; Example: `./file_batcher_cache`; Default: None
            - file_type (str): File type; Example: JSON; Default: JSON
            - filter_nulls (bool): Whether to filter null values; Example: false; Default: false
            - schema (dict): Data schema; See below; Default: `[Required]`
            - timestamp_column_name (str): Name of the timestamp column; Example: timestamp; Default: timestamp

        batching_options:
            - end_time (datetime/string): Endtime of the time window; Example: "2023-03-14T23:59:59"; Default: None
            - iso_date_regex_pattern (str): Regex pattern for ISO date matching;
                Example: "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}"; Default: <iso_date_regex_pattern>
            - parser_kwargs (dict): Additional arguments for the parser; Example: {}; Default: {}
            - period (str): Time period for grouping files; Example: "1d"; Default: "1d"
            - sampling_rate_s (int): Sampling rate in seconds; Example: 0; Default: None
            - sampling (int): Sampling rate in seconds; Example: 0; Default: None
            - start_time (datetime/string): Start time of the time window; Example: "2023-03-01T00:00:00"; Default: None

        schema:
            - encoding (str): Encoding; Example: "latin1"; Default: "latin1"
            - schema_str (str): Schema string; Example: "string"; Default: `[Required]`
    """

    config = builder.get_current_module_config()
    sampling = config.get("sampling", None)
    sampling_rate_s = config.get("sampling_rate_s", None)

    iso_date_regex_pattern = config.get("iso_date_regex_pattern", DEFAULT_ISO_DATE_REGEX_PATTERN)
    iso_date_regex = re.compile(iso_date_regex_pattern)

    if (sampling_rate_s is not None and sampling_rate_s > 0):
        assert sampling is None, "Cannot set both sampling and sampling_rate_s at the same time"

        # Show the deprecation message
        warnings.warn(("The `sampling_rate_s` argument has been deprecated. "
                       "Please use `sampling={sampling_rate_s}S` instead"),
                      DeprecationWarning)

        sampling = f"{sampling_rate_s}S"

    default_batching_opts = {
        "period": config.get("period", 'D'),
        "sampling_rate_s": sampling_rate_s,
        "sampling": sampling,
        "start_time": config.get("start_time"),
        "end_time": config.get("end_time"),
        "iso_date_regex_pattern": iso_date_regex_pattern
    }

    default_file_to_df_opts = {
        "timestamp_column_name": config.get("timestamp_column_name"),
        "schema": config.get("schema"),
        "file_type": config.get("file_type"),
        "filter_null": config.get("filter_null"),
        "parser_kwargs": config.get("parser_kwargs"),
        "cache_dir": config.get("cache_dir")
    }

    def validate_control_message(control_message: ControlMessage):
        if control_message.has_metadata("batching_options") and not isinstance(
                control_message.get_metadata("batching_options"), dict):
            raise ValueError("Invalid or missing 'batching_options' metadata in control message")

        data_type = control_message.get_metadata("data_type")
        if data_type not in {"payload", "streaming"}:
            raise ValueError(f"Invalid 'data_type' metadata in control message: {data_type}")

    def build_period_batches(files: typing.List[str],
                             params: typing.Dict[any, any]) -> typing.List[typing.Tuple[typing.List[str], int]]:
        file_objects: fsspec.core.OpenFiles = fsspec.open_files(files)

        nonlocal iso_date_regex_pattern
        nonlocal iso_date_regex

        if params["iso_date_regex_pattern"] != iso_date_regex_pattern:
            iso_date_regex_pattern = params["iso_date_regex_pattern"]
            iso_date_regex = re.compile(iso_date_regex_pattern)

        try:
            start_time = params["start_time"]
            end_time = params["end_time"]
            period = params["period"]
            sampling_rate_s = params["sampling_rate_s"]
            sampling = params["sampling"]

            if not isinstance(start_time, (str, type(None))) or (start_time is not None
                                                                 and not re.match(r"\d{4}-\d{2}-\d{2}", start_time)):
                raise ValueError(f"Invalid 'start_time' value: {start_time}")

            if not isinstance(end_time, (str, type(None))) or (end_time is not None
                                                               and not re.match(r"\d{4}-\d{2}-\d{2}", end_time)):
                raise ValueError(f"Invalid 'end_time' value: {end_time}")

            if (sampling_rate_s is not None and sampling_rate_s > 0):
                assert sampling is None, "Cannot set both sampling and sampling_rate_s at the same time"

                # Show the deprecation message
                warnings.warn(("The `sampling_rate_s` argument has been deprecated. "
                               "Please use `sampling={sampling_rate_s}S` instead"),
                              DeprecationWarning)

                sampling = f"{sampling_rate_s}S"

            if (start_time is not None):
                start_time = datetime.datetime.fromisoformat(start_time).replace(tzinfo=datetime.timezone.utc)

            if (end_time is not None):
                end_time = datetime.datetime.fromisoformat(end_time).replace(tzinfo=datetime.timezone.utc)

        except Exception as exec_info:
            logger.error("Error parsing parameters: %s", (exec_info))
            raise

        timestamps = []
        full_names = []

        for file_object in file_objects:
            time_stamp = date_extractor(file_object, iso_date_regex)

            # Exclude any files outside the time window
            if ((start_time is not None and time_stamp < start_time)
                    or (end_time is not None and time_stamp > end_time)):
                continue

            timestamps.append(time_stamp)
            full_names.append(file_object.full_name)

        # Build the dataframe
        df = pd.DataFrame(index=pd.DatetimeIndex(timestamps), data={"filename": full_names})

        # sort the incoming data by date
        df.sort_index(inplace=True)

        # If sampling was provided, perform that here
        if (sampling is not None):

            if (isinstance(sampling, str)):
                # We have a frequency for sampling. Resample by the frequency, taking the first
                df = df.resample(sampling).first().dropna()

            elif (sampling < 1.0):
                # Sample a fraction of the rows
                df = df.sample(frac=sampling).sort_index()

            else:
                # Sample a fixed amount
                df = df.sample(n=sampling).sort_index()

        # Early exit if no files were found
        if (len(df) == 0):
            return []

        if (period is None):
            # No period was set so group them all into one single batch
            return [(df["filename"].to_list(), len(df))]

        # Now group the rows by the period
        resampled = df.resample(period)

        n_groups = len(resampled)

        output_batches = []

        for _, period_df in resampled:

            filename_list = period_df["filename"].to_list()

            output_batches.append((filename_list, n_groups))

        return output_batches

    def build_file_df_params(control_message: ControlMessage) -> typing.Dict[any, any]:
        file_to_df_opts = {}
        if (control_message.has_metadata("file_to_df_options")):
            file_to_df_opts = control_message.get_metadata("file_to_df_options")

        return merge_dictionaries(file_to_df_opts, default_file_to_df_opts)

    def generate_cms_for_batch_periods(
            control_message: ControlMessage,
            batch_periods: typing.List[typing.Tuple[typing.List[str], int]]) -> typing.List[ControlMessage]:
        data_type = control_message.get_metadata("data_type")
        file_to_df_params = build_file_df_params(control_message=control_message)

        control_messages = []

        for batch_period in batch_periods:

            filenames = batch_period[0]
            n_groups = batch_period[1]

            if filenames:
                load_task = {
                    "loader_id": FILE_TO_DF_LOADER,
                    "strategy": "aggregate",
                    "files": filenames,
                    "n_groups": n_groups,
                    "batcher_config": {  # TODO(Devin): Remove this when we're able to attach config to the loader
                        "timestamp_column_name": file_to_df_params.get("timestamp_column_name"),
                        "schema": file_to_df_params.get("schema"),
                        "file_type": file_to_df_params.get("file_type"),
                        "filter_null": file_to_df_params.get("filter_null"),
                        "parser_kwargs": file_to_df_params.get("parser_kwargs"),
                        "cache_dir": file_to_df_params.get("cache_dir")
                    }
                }

                if (data_type in ("payload", "streaming")):
                    batch_control_message = control_message.copy()
                    batch_control_message.add_task("load", load_task)
                    control_messages.append(batch_control_message)
                else:
                    raise ValueError(f"Unknown data type: {data_type}")

        return control_messages

    def build_processing_params(control_message) -> typing.Dict[any, any]:
        batching_opts = {}
        if (control_message.has_metadata("batching_options")):
            batching_opts = control_message.get_metadata("batching_options")

        return merge_dictionaries(batching_opts, default_batching_opts)

    def on_data(control_message: ControlMessage) -> typing.List[ControlMessage]:
        try:
            validate_control_message(control_message)

            message_meta = control_message.payload()

            params = build_processing_params(control_message)
            with message_meta.mutable_dataframe() as dfm:
                files = dfm.files.to_arrow().to_pylist()
                batch_periods = build_period_batches(files, params)

            control_messages = generate_cms_for_batch_periods(control_message, batch_periods)

            return control_messages

        except Exception as exec_info:
            logger.error("Error building file list, discarding control message %s", exec_info)
            return []

    def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):
        obs.pipe(ops.map(on_data), ops.flatten()).subscribe(sub)

    node = builder.make_node(FILE_BATCHER, mrc.core.operators.build(node_fn))

    # Register input and output port for a module.
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
