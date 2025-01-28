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

import collections.abc
import logging
import time
from io import StringIO

import mrc
import pandas as pd
import pika

from morpheus.config import ExecutionMode
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline.stage_decorator import source
from morpheus.utils.type_utils import get_df_pkg

logger = logging.getLogger(__name__)


@source(name="from-rabbitmq", execution_modes=(ExecutionMode.GPU, ExecutionMode.CPU))
def rabbitmq_source(subscription: mrc.Subscription,
                    host: str,
                    exchange: str,
                    exchange_type: str = 'fanout',
                    queue_name: str = '',
                    poll_interval: str = '100millis') -> collections.abc.Iterator[MessageMeta]:
    """
    Source stage used to load messages from a RabbitMQ queue.

    Parameters
    ----------
    subscription : mrc.Subscription
        Subscription object used to determine if the pipeline is still running.
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
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))

    channel = connection.channel()
    channel.exchange_declare(exchange=exchange, exchange_type=exchange_type)

    result = channel.queue_declare(queue=queue_name, exclusive=True)

    # When queue_name='' we will receive a randomly generated queue name
    queue_name = result.method.queue

    channel.queue_bind(exchange=exchange, queue=queue_name)

    poll_interval = pd.Timedelta(poll_interval)

    df_pkg = get_df_pkg()

    try:
        while subscription.is_subscribed():
            (method_frame, _, body) = channel.basic_get(queue_name)
            if method_frame is not None:
                try:
                    buffer = StringIO(body.decode("utf-8"))
                    df = df_pkg.read_json(buffer, orient='records', lines=True)
                    yield MessageMeta(df=df)
                except Exception as ex:
                    logger.exception("Error occurred converting RabbitMQ message to Dataframe: %s", ex)
                finally:
                    channel.basic_ack(method_frame.delivery_tag)
            else:
                # queue is empty, sleep before polling again
                time.sleep(poll_interval.total_seconds())

    finally:
        connection.close()
