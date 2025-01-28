# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from morpheus.common import TypeId
from morpheus.messages import MessageMeta
from morpheus.pipeline.stage_decorator import stage


@stage(
    needed_columns={
        'to_count': TypeId.INT32,
        'bcc_count': TypeId.INT32,
        'cc_count': TypeId.INT32,
        'total_recipients': TypeId.INT32,
        'data': TypeId.STRING
    })
def recipient_features_stage(message: MessageMeta, *, sep_token: str = '[SEP]') -> MessageMeta:
    # Open the DataFrame from the incoming message for in-place modification
    with message.mutable_dataframe() as df:
        df['to_count'] = df['To'].str.count('@')
        df['bcc_count'] = df['BCC'].str.count('@')
        df['cc_count'] = df['CC'].str.count('@')
        df['total_recipients'] = df['to_count'] + df['bcc_count'] + df['cc_count']

        # Attach features to string data
        df['data'] = (df['to_count'].astype(str) + sep_token + df['bcc_count'].astype(str) + sep_token +
                      df['cc_count'].astype(str) + sep_token + df['total_recipients'].astype(str) + sep_token +
                      df['Message'])

    # Return the message for the next stage
    return message
