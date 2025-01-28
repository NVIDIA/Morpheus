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
from functools import partial

import mrc

from morpheus.controllers.serialize_controller import SerializeController
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_ids import SERIALIZE
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)


@register_module(SERIALIZE, MORPHEUS_MODULE_NAMESPACE)
def serialize(builder: mrc.Builder):
    """
    Includes & excludes columns from messages.

    This module filters columns from a `ControlMessage` object emitting a `MessageMeta`.

    Parameters
    ----------
    builder : mrc.Builder
        mrc Builder object.

    Notes
    -----
        Configurable Parameters:
            - columns (list[str]): List of columns to include; Example: `["column1", "column2", "column3"]`;
                Default: None
            - exclude (list[str]): List of regex patterns to exclude columns; Example: `["column_to_exclude"]`;
                Default: `[r'^ID$', r'^_ts_']`
            - fixed_columns (bool): If true, the columns are fixed and not determined at runtime; Example: `true`;
                Default: true
            - include (str): Regex to include columns; Example: `^column`; Default: None
            - use_cpp (bool): If true, use C++ to serialize; Example: `true`; Default: false
    """

    config = builder.get_current_module_config()

    include = config.get("include", None)
    exclude = config.get("exclude", [r'^ID$', r'^_ts_'])
    fixed_columns = config.get("fixed_columns", True)

    controller = SerializeController(include=include, exclude=exclude, fixed_columns=fixed_columns)

    include_columns = controller.get_include_col_pattern()
    exclude_columns = controller.get_exclude_col_pattern()

    node = builder.make_node(
        SERIALIZE, partial(controller.convert_to_df, include_columns=include_columns, exclude_columns=exclude_columns))

    # Register input and output port for a module.
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
