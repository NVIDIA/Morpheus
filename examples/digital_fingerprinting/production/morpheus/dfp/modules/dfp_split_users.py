# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
import typing

import mrc
import numpy as np
import pandas as pd
from dfp.utils.logging_timer import log_time
from mrc.core import operators as ops

import cudf

from morpheus.messages import MessageControl
from morpheus.messages import MessageMeta
from morpheus.utils.module_ids import MODULE_NAMESPACE
from morpheus.utils.module_utils import get_module_config
from morpheus.utils.module_utils import register_module

from ..utils.module_ids import DFP_SPLIT_USERS

logger = logging.getLogger("morpheus.{}".format(__name__))


@register_module(DFP_SPLIT_USERS, MODULE_NAMESPACE)
def dfp_split_users(builder: mrc.Builder):
    """
    This module function split the data based on user Id's.

    Parameters
    ----------
    builder : mrc.Builder
        Pipeline budler instance.
    """

    config = get_module_config(DFP_SPLIT_USERS, builder)

    skip_users = config.get("skip_users", [])
    only_users = config.get("only_users", [])
    timestamp_column_name = config.get("timestamp_column_name", None)
    userid_column_name = config.get("userid_column_name", None)
    fallback_username = config.get("fallback_username", None)
    include_generic = config.get("include_generic", False)
    include_individual = config.get("include_individual", False)

    # Map of user ids to total number of messages. Keep indexes monotonic and increasing per user
    user_index_map: typing.Dict[str, int] = {}

    def extract_users(control_message: MessageControl):
        logger.debug("Extracting users from message")
        if (control_message is None):
            logger.debug("No message to extract users from")
            return []

        mm = control_message.payload()
        with mm.mutable_dataframe() as dfm:
            # df = control_message.payload().df
            with log_time(logger.debug) as log_info:

                if (isinstance(dfm, cudf.DataFrame)):
                    # Convert to pandas because cudf is slow at this
                    df = dfm.to_pandas()
                    df[timestamp_column_name] = pd.to_datetime(df[timestamp_column_name], utc=True)

                split_dataframes: typing.Dict[str, cudf.DataFrame] = {}

                # If we are skipping users, do that here
                if (len(skip_users) > 0):
                    df = df[~df[userid_column_name].isin(skip_users)]

                if (len(only_users) > 0):
                    df = df[df[userid_column_name].isin(only_users)]

                # Split up the dataframes
                if (include_generic):
                    split_dataframes[fallback_username] = df

                if (include_individual):
                    split_dataframes.update(
                        {username: user_df
                         for username, user_df in df.groupby("username", sort=False)})

                output_messages: typing.List[MessageControl] = []

                for user_id in sorted(split_dataframes.keys()):
                    if (user_id in skip_users):
                        continue

                    user_df = split_dataframes[user_id]

                    current_user_count = user_index_map.get(user_id, 0)
                    logger.debug("Current user count: %s", current_user_count)

                    # Reset the index so that users see monotonically increasing indexes
                    user_df.index = range(current_user_count, current_user_count + len(user_df))
                    user_index_map[user_id] = current_user_count + len(user_df)

                    user_control_message = control_message.copy()
                    user_control_message.set_metadata("user_id", user_id)

                    user_cudf = cudf.from_pandas(user_df)
                    user_control_message.payload(MessageMeta(df=user_cudf))

                    # output_messages.append(DFPMessageMeta(df=user_df, user_id=user_id))
                    output_messages.append(user_control_message)

                    rows_per_user = [len(msg.payload().df.to_pandas()) for msg in output_messages]

                    if (len(output_messages) > 0):
                        log_info.set_log(
                            ("Batch split users complete. Input: %s rows from %s to %s. "
                             "Output: %s users, rows/user min: %s, max: %s, avg: %.2f. Duration: {duration:.2f} ms"),
                            len(df),
                            df[timestamp_column_name].min(),
                            df[timestamp_column_name].max(),
                            len(rows_per_user),
                            np.min(rows_per_user),
                            np.max(rows_per_user),
                            np.mean(rows_per_user),
                        )

                return output_messages

    def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):
        obs.pipe(ops.map(extract_users), ops.flatten()).subscribe(sub)

    node = builder.make_node(DFP_SPLIT_USERS, mrc.core.operators.build(node_fn))

    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
