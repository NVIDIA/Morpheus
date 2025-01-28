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
import time
from io import StringIO

import mrc
import pandas as pd
import pika

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stage_schema import StageSchema

logger = logging.getLogger(__name__)


@register_stage("from-rabbitmq")
class RabbitMQSourceStage(PreallocatorMixin, GpuAndCpuMixin, SingleOutputSource):
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
    queue_name : str, optional
        Name of the queue to listen to. If left blank, RabbitMQ will generate a random queue name bound to the exchange.
    poll_interval : str, optional
        Amount of time  between polling RabbitMQ for new messages
    """

    def __init__(self,
                 config: Config,
                 host: str,
                 exchange: str,
                 exchange_type: str = 'fanout',
                 queue_name: str = '',
                 poll_interval: str = '100millis'):
        super().__init__(config)
        self._host = host
        self._exchange = exchange
        self._exchange_type = exchange_type
        self._queue_name = queue_name

        self._connection: pika.BlockingConnection = None
        self._channel = None

        self._poll_interval = pd.Timedelta(poll_interval)

    @property
    def name(self) -> str:
        return "from-rabbitmq"

    def supports_cpp_node(self) -> bool:
        return True

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(MessageMeta)

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:
        if self._build_cpp_node():
            from ._lib import rabbitmq_cpp_stage

            node = rabbitmq_cpp_stage.RabbitMQSourceStage(builder,
                                                          self.unique_name,
                                                          self._host,
                                                          self._exchange,
                                                          self._exchange_type,
                                                          self._queue_name,
                                                          self._poll_interval.to_pytimedelta())
        else:
            self.connect()
            node = builder.make_source(self.unique_name, self.source_generator)

        return node

    def connect(self):
        self._connection = pika.BlockingConnection(pika.ConnectionParameters(host=self._host))

        self._channel = self._connection.channel()
        self._channel.exchange_declare(exchange=self._exchange, exchange_type=self._exchange_type)

        result = self._channel.queue_declare(queue=self._queue_name, exclusive=True)

        # When queue_name='' we will receive a randomly generated queue name
        self._queue_name = result.method.queue

        self._channel.queue_bind(exchange=self._exchange, queue=self._queue_name)

    def source_generator(self, subscription: mrc.Subscription):
        try:
            while not self.is_stop_requested() and subscription.is_subscribed():
                (method_frame, _, body) = self._channel.basic_get(self._queue_name)
                if method_frame is not None:
                    try:
                        buffer = StringIO(body.decode("utf-8"))
                        df = pd.read_json(buffer, orient='records', lines=True)
                        yield MessageMeta(df=df)
                    except Exception as ex:
                        logger.exception("Error occurred converting RabbitMQ message to Dataframe: %s", ex)
                    finally:
                        self._channel.basic_ack(method_frame.delivery_tag)
                else:
                    # queue is empty, sleep before polling again
                    time.sleep(self._poll_interval.total_seconds())

        finally:
            self._connection.close()
