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

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage


@register_stage("recipient-features", modes=[PipelineModes.NLP])
class RecipientFeaturesStage(PassThruTypeMixin, SinglePortStage):
    """
    Pre-processing stage which counts the number of recipients in an email's metadata.

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    sep_token : str
        Bert separator token.
    """

    def __init__(self, config: Config, sep_token: str = '[SEP]'):
        super().__init__(config)
        if config.mode != PipelineModes.NLP:
            raise RuntimeError("RecipientFeaturesStage must be used in a pipeline configured for NLP")

        if len(sep_token) > 0:
            self._sep_token = sep_token
        else:
            raise ValueError("sep_token cannot be an empty string")

        # This stage adds new columns to the DataFrame, as an optimization we define the columns that are needed,
        # ensuring that these columns are pre-allocated with null values. This action is performed by Morpheus for any
        # stage defining this attribute.
        self._needed_columns.update({
            'to_count': TypeId.INT32,
            'bcc_count': TypeId.INT32,
            'cc_count': TypeId.INT32,
            'total_recipients': TypeId.INT32,
            'data': TypeId.STRING
        })

    @property
    def name(self) -> str:
        return "recipient-features"

    def accepted_types(self) -> tuple:
        return (MessageMeta, )

    def supports_cpp_node(self) -> bool:
        return False

    def on_data(self, message: MessageMeta) -> MessageMeta:
        # Open the DataFrame from the incoming message for in-place modification
        with message.mutable_dataframe() as df:
            df['to_count'] = df['To'].str.count('@')
            df['bcc_count'] = df['BCC'].str.count('@')
            df['cc_count'] = df['CC'].str.count('@')
            df['total_recipients'] = df['to_count'] + df['bcc_count'] + df['cc_count']

            # Attach features to string data
            df['data'] = (df['to_count'].astype(str) + self._sep_token + df['bcc_count'].astype(str) + self._sep_token +
                          df['cc_count'].astype(str) + self._sep_token + df['total_recipients'].astype(str) +
                          self._sep_token + df['Message'])

        # Return the message for the next stage
        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
