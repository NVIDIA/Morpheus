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
"""Morpheus pipeline module for fetching files and emitting them as DataFrames."""

import logging
import pickle

import mrc
from mrc.core import operators as ops

from morpheus.cli.utils import str_to_file_type
from morpheus.controllers.file_to_df_controller import FileToDFController
from morpheus.utils.module_ids import FILE_TO_DF
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)


@register_module(FILE_TO_DF, MORPHEUS_MODULE_NAMESPACE)
def file_to_df(builder: mrc.Builder):
    """
    This module reads data from batched files into a DataFrame after receiving input from the "FileBatcher" module.
    It can load file content from both local disk and S3 buckets.

    Parameters
    ----------
    builder : mrc.Builder
        mrc Builder object.

    Notes
    -----
    Configurable parameters:
        - cache_dir (str): Directory to cache the rolling window data.
        - file_type (str): Type of the input file.
        - filter_null (bool): Whether to filter out null values.
        - parser_kwargs (dict): Keyword arguments to pass to the parser.
        - schema (dict): Schema of the input data.
        - timestamp_column_name (str): Name of the timestamp column.
    """

    config = builder.get_current_module_config()

    timestamp_column_name = config.get("timestamp_column_name", "timestamp")

    if ("schema" not in config) or (config["schema"] is None):
        raise ValueError("Input schema is required.")

    schema_config = config["schema"]
    schema_str = schema_config["schema_str"]
    encoding = schema_config["encoding"]

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

    controller = FileToDFController(schema=schema,
                                    filter_null=filter_null,
                                    file_type=file_type,
                                    parser_kwargs=parser_kwargs,
                                    cache_dir=cache_dir,
                                    timestamp_column_name=timestamp_column_name)

    node = builder.make_node(FILE_TO_DF, ops.map(controller.convert_to_dataframe), ops.on_completed(controller.close))

    # Register input and output port for a module.
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
