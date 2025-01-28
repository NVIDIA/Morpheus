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

import logging
from io import StringIO

import mrc
import pika

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage

logger = logging.getLogger(__name__)


@register_stage("to-rabbitmq")
class WriteToRabbitMQStage(PassThruTypeMixin, GpuAndCpuMixin, SinglePortStage):
    """
    Source stage used to load messages from a RabbitMQ queue.

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    host : str
        Hostname or IP of the RabbitMQ server.
    exchange : str
        Name of the RabbitMQ exchange to connect to.
    exchange_type : str, optional
        RabbitMQ exchange type; defaults to `fanout`.
    routing_key : str, optional
        RabbitMQ routing key if needed.
    """

    def __init__(self, config: Config, host: str, exchange: str, exchange_type: str = 'fanout', routing_key: str = ''):
        super().__init__(config)
        self._connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))

        self._exchange = exchange
        self._routing_key = routing_key

        self._channel = self._connection.channel()
        self._channel.exchange_declare(exchange=self._exchange, exchange_type=exchange_type)

    @property
    def name(self) -> str:
        return "to-rabbitmq"

    def accepted_types(self) -> tuple:
        return (MessageMeta, )

    def supports_cpp_node(self) -> bool:
        return False

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_sink(self.unique_name, self.on_data, self.on_error, self.on_complete)
        builder.make_edge(input_node, node)
        return node

    def on_data(self, message: MessageMeta) -> MessageMeta:
        df = message.df

        buffer = StringIO()
        df.to_json(buffer, orient='records', lines=True)
        body = buffer.getvalue().strip()

        self._channel.basic_publish(exchange=self._exchange, routing_key=self._routing_key, body=body)

        return message

    def on_error(self, ex: Exception):
        logger.exception("Error occurred : %s", ex)
        self._connection.close()

    def on_complete(self):
        self._connection.close()
