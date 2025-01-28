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
"""Loader for fetching files and emitting them as DataFrames."""

import logging
import pickle

import fsspec

import cudf

from morpheus.cli.utils import str_to_file_type
from morpheus.controllers.file_to_df_controller import FileToDFController
from morpheus.messages import ControlMessage
from morpheus.messages.message_meta import MessageMeta
from morpheus.utils.downloader import DownloadMethods
from morpheus.utils.loader_ids import FILE_TO_DF_LOADER
from morpheus.utils.loader_utils import register_loader

logger = logging.getLogger(__name__)


@register_loader(FILE_TO_DF_LOADER)
def file_to_df_loader(control_message: ControlMessage, task: dict):
    """
    This function is used to load files containing data into a dataframe. Dataframe is created by
    processing files either using a single thread, dask, or dask_thread. This function determines
    the download method to use, and if it starts with "dask," it creates a dask client and uses it to process the files.
    Otherwise, it uses a single thread to process the files. This function then caches the resulting
    dataframe using a hash of the file paths. The dataframe is wrapped in a MessageMeta and then attached as a payload
    to a ControlMessage object and passed on to further stages.

    Parameters
    ----------
    control_message : ControlMessage
        The ControlMessage object containing the pipeline control message.
    task : typing.Dict[any, any]
        A dictionary representing the current task in the pipeline control message.

    Returns
    -------
    message : ControlMessage
        Updated message control object with payload as a MessageMeta.

    Raises
    ------
    RuntimeError:
        If no files matched the input strings specified in the task, or if there was an error loading the data.
    """
    if task.get("strategy", "aggregate") != "aggregate":
        raise RuntimeError("Only 'aggregate' strategy is supported for file_to_df loader.")

    files = task.get("files", None)
    n_groups = task.get("n_groups", None)

    config = task["batcher_config"]

    timestamp_column_name = config.get("timestamp_column_name", "timestamp")

    if ("schema" not in config) or (config["schema"] is None):
        raise ValueError("Input schema is required.")

    schema_config = config["schema"]
    schema_str = schema_config["schema_str"]
    encoding = schema_config.get("encoding", 'latin1')

    file_type = config.get("file_type", "JSON")
    filter_null = config.get("filter_null", False)
    parser_kwargs = config.get("parser_kwargs", None)
    cache_dir = config.get("cache_dir", None)

    if (cache_dir is None):
        cache_dir = "./.cache"
        logger.warning("Cache directory not set. Defaulting to ./.cache")

    # Load input schema
    schema = pickle.loads(bytes(schema_str, encoding))

    try:
        file_type = str_to_file_type(file_type.lower())
    except Exception as exec_info:
        raise ValueError(f"Invalid input file type '{file_type}'. Available file types are: CSV, JSON.") from exec_info

    try:
        controller = FileToDFController(schema=schema,
                                        filter_null=filter_null,
                                        file_type=file_type,
                                        parser_kwargs=parser_kwargs,
                                        cache_dir=cache_dir,
                                        timestamp_column_name=timestamp_column_name,
                                        download_method=DownloadMethods.SINGLE_THREAD)
        pdf = controller.convert_to_dataframe(file_object_batch=(fsspec.open_files(files), n_groups))
        df = cudf.from_pandas(pdf)

        # Overwriting payload with derived data
        control_message.payload(MessageMeta(df))

    finally:
        controller.close()

    return control_message
