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

import pandas as pd

from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.type_utils import is_cudf_type


def concat_dataframes(messages: list[ControlMessage | MessageMeta]) -> pd.DataFrame:
    """
    Concatinate the DataFrame associated with the collected messages into a single Pandas DataFrame.

    Parameters
    ----------
    messages : list[ControlMessage] | list[MessageMeta]
        Messages containing DataFrames to concat.

    Returns
    -------
    pd.DataFrame
    """

    all_meta = []
    for msg in messages:
        if isinstance(msg, ControlMessage):
            df = msg.payload().df
        elif isinstance(msg, MessageMeta):
            df = msg.df
        else:
            raise ValueError("Invalid message type")

        if is_cudf_type(df):
            df = df.to_pandas()

        all_meta.append(df)

    return pd.concat(all_meta)
