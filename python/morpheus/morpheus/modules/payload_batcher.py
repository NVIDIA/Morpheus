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
import warnings

import mrc
from mrc.core import operators as ops

from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.control_message_utils import cm_default_failure_context_manager
from morpheus.utils.control_message_utils import cm_skip_processing_if_failed
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_ids import PAYLOAD_BATCHER
from morpheus.utils.module_utils import register_module
from morpheus.utils.type_aliases import DataFrameType
from morpheus.utils.type_utils import get_df_pkg_from_obj
from morpheus.utils.type_utils import is_cudf_type

logger = logging.getLogger(__name__)


@register_module(PAYLOAD_BATCHER, MORPHEUS_MODULE_NAMESPACE)
def payload_batcher(builder: mrc.Builder):
    """
    Batches incoming control message data payload into smaller batches based on the specified configurations.

    Parameters
    ----------
    builder : mrc.Builder
        The mrc Builder object used to configure the module.

    Notes
    -----
    Configurable Parameters:
        - max_batch_size (int): The maximum size of each batch (default: 256).
        - raise_on_failure (bool): Whether to raise an exception if a failure occurs during processing (default: False).
        - group_by_columns (list): The column names to group by when batching (default: []).
        - disable_max_batch_size (bool): Whether to disable the max_batch_size and only batch by group (default: False).
        - timestamp_column_name (str): The name of the timestamp column (default: None).
        - timestamp_pattern (str): The pattern to parse the timestamp column (default: None).
        - period (str): The period for grouping by timestamp (default: 'D').

    Raises
    ------
    ValueError
        If disable_max_batch_size is True and group_by_columns is empty or None.

    """

    config = builder.get_current_module_config()
    max_batch_size = config.get("max_batch_size", 256)
    raise_on_failure = config.get("raise_on_failure", False)
    group_by_columns = config.get("group_by_columns", [])
    disable_max_batch_size = config.get("disable_max_batch_size", False)
    timestamp_column_name = config.get("timestamp_column_name", None)
    timestamp_pattern = config.get("timestamp_pattern", None)
    period = config.get("period", "D")
    has_timestamp_column = False
    period_column = "period"

    if disable_max_batch_size and not (group_by_columns or timestamp_column_name):
        raise ValueError("When disable_max_batch_size is True and group_by_columns must not be empty or None.")

    # Check if a timestamp column name is provided
    if timestamp_column_name:
        for idx, column in enumerate(group_by_columns):
            # Check if the current column matches the timestamp column name
            if timestamp_column_name == column:
                # Check if the timestamp pattern is not specified
                if timestamp_pattern is None:
                    warnings.warn("Timestamp column name is provided, but the timestamp pattern is not specified.")
                # Remove the current column from the group_by_columns list
                group_by_columns.pop(idx)
                # Insert the period column at the same index
                group_by_columns.insert(idx, period_column)
                # Set the flag indicating the presence of the timestamp column
                has_timestamp_column = True
                # Exit the loop
                break

        # Check if group_by_columns is empty or if it doesn't contain the timestamp column
        if (not group_by_columns) or (group_by_columns and not has_timestamp_column):
            # Check if the timestamp pattern is not specified
            if timestamp_pattern is None:
                warnings.warn("Timestamp column name is provided, but the timestamp pattern is not specified.")
            # Set the flag indicating the presence of the timestamp column
            has_timestamp_column = True
            # Add the period column to the group_by_columns
            group_by_columns.append(period_column)

    @cm_skip_processing_if_failed
    @cm_default_failure_context_manager(raise_on_failure=raise_on_failure)
    def on_next(control_message: ControlMessage) -> list[ControlMessage]:
        nonlocal disable_max_batch_size

        message_meta = control_message.payload()
        control_messages = []
        with message_meta.mutable_dataframe() as dfm:
            dfs = _batch_dataframe(dfm) if not disable_max_batch_size else _batch_dataframe_by_group(dfm)
            logger.debug("Number of batches created: %s", len(dfs))
            for df in dfs:
                ret_cm = control_message.copy()
                df = df.reset_index(drop=True)
                ret_cm.payload(MessageMeta(df))
                control_messages.append(ret_cm)

        return control_messages

    def _batch_dataframe(df: DataFrameType) -> list[DataFrameType]:
        nonlocal max_batch_size

        dfm_length = len(df)
        dfs = []
        if dfm_length <= max_batch_size:
            dfs.append(df)
        else:
            num_batches = (dfm_length + max_batch_size - 1) // max_batch_size
            dfs = [df.iloc[i * max_batch_size:(i + 1) * max_batch_size] for i in range(num_batches)]
        return dfs

    def _batch_dataframe_by_group(df: DataFrameType) -> list[DataFrameType]:
        nonlocal max_batch_size
        nonlocal group_by_columns
        nonlocal timestamp_column_name
        nonlocal timestamp_pattern
        nonlocal has_timestamp_column
        nonlocal period_column
        nonlocal period

        if has_timestamp_column:

            # Apply timestamp pattern and group by the formatted timestamp column
            df_pkg = get_df_pkg_from_obj(df)
            period_series = df_pkg.to_datetime(df[timestamp_column_name], format=timestamp_pattern)

            if is_cudf_type(df):
                # Period object conversion is not supported in cudf
                period_series = period_series.to_pandas()

            df[period_column] = period_series.dt.to_period(period).astype('str')

        if len(group_by_columns) == 1:
            # Avoid warning from cudf regardning an upcoming change of behavior when applying a groupby to a single
            # element list.
            group_by_columns_ = group_by_columns[0]
        else:
            group_by_columns_ = group_by_columns

        groups = df.groupby(group_by_columns_)

        dfs = []
        for _, group in groups:
            if disable_max_batch_size:
                dfs.append(group)
            else:
                group_length = len(group)
                if group_length <= max_batch_size:
                    dfs.append(group)
                else:
                    num_batches = (group_length + max_batch_size - 1) // max_batch_size
                    group_batches = [
                        group.iloc[i * max_batch_size:(i + 1) * max_batch_size] for i in range(num_batches)
                    ]
                    dfs.extend(group_batches)

        return dfs

    node = builder.make_node("internal_node", ops.map(on_next), ops.flatten())

    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
