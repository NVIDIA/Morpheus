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

import json
import logging
import time
import typing

import confluent_kafka as ck
import mrc
import pandas as pd

from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.stages.input.kafka_source_stage import AutoOffsetReset

logger = logging.getLogger(__name__)


class ControlMessageKafkaSourceStage(PreallocatorMixin, SingleOutputSource):
    """
    Load control messages from a Kafka cluster.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    bootstrap_servers : str
        Comma-separated list of bootstrap servers. If using Kafka created via `docker-compose`, this can be set to
        'auto' to automatically determine the cluster IPs and ports
    input_topic : typing.List[str], default = ["test_cm"]
        Name of the Kafka topic from which messages will be consumed. To consume from multiple topics,
        repeat the same option multiple times.
    group_id : str
        Specifies the name of the consumer group a Kafka consumer belongs to.
    client_id : str, default = None
        An optional identifier of the consumer.
    poll_interval : str
        Seconds that elapse between polling Kafka for new messages. Follows the pandas interval format.
    disable_commit : bool, default = False
        Enabling this option will skip committing messages as they are pulled off the server. This is only useful for
        debugging, allowing the user to process the same messages multiple times.
    disable_pre_filtering : bool, default = False
        Enabling this option will skip pre-filtering of json messages. This is only useful when inputs are known to be
        valid json.
    auto_offset_reset : `AutoOffsetReset`, case_sensitive = False
        Sets the value for the configuration option 'auto.offset.reset'. See the kafka documentation for more
        information on the effects of each value."
    stop_after: int, default = 0
        Stops ingesting after emitting `stop_after` records (rows in the dataframe). Useful for testing. Disabled if `0`
    async_commits: bool, default = True
        Enable commits to be performed asynchronously. Ignored if `disable_commit` is `True`.
    """

    def __init__(self,
                 c: Config,
                 bootstrap_servers: str,
                 input_topic: typing.List[str] = None,
                 group_id: str = "morpheus",
                 client_id: str = None,
                 poll_interval: str = "10millis",
                 disable_commit: bool = False,
                 disable_pre_filtering: bool = False,
                 auto_offset_reset: AutoOffsetReset = AutoOffsetReset.LATEST,
                 stop_after: int = 0,
                 async_commits: bool = True):

        super().__init__(c)

        if isinstance(auto_offset_reset, AutoOffsetReset):
            auto_offset_reset = auto_offset_reset.value

        self._consumer_params = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'session.timeout.ms': "60000",
            "auto.offset.reset": auto_offset_reset
        }
        if client_id is not None:
            self._consumer_params['client.id'] = client_id

        input_topic = input_topic if (input_topic is not None) else ["test_cm"]
        if isinstance(input_topic, str):
            input_topic = [input_topic]

        # Remove duplicate topics if there are any.
        self._topics = list(set(input_topic))

        self._max_concurrent = c.num_threads
        self._disable_commit = disable_commit
        self._disable_pre_filtering = disable_pre_filtering
        self._stop_after = stop_after
        self._async_commits = async_commits
        self._client = None

        self._poll_interval = pd.Timedelta(poll_interval).total_seconds()
        self._started = False

        self._records_emitted = 0
        self._num_messages = 0

    @property
    def name(self) -> str:
        return "from-cm-kafka"

    def supports_cpp_node(self):
        return False

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(ControlMessage)

    def _process_msg(self, consumer, msg):
        control_messages = []

        payload = msg.value()
        if payload is not None:
            try:
                decoded_msg = payload.decode("utf-8")
                control_messages_conf = json.loads(decoded_msg)

                self._num_messages += 1
                # TODO(Devin) - one CM at a time(?), don't need to submit 'inputs'
                for control_message_conf in control_messages_conf.get("inputs", []):
                    self._records_emitted += 1
                    control_messages.append(ControlMessage(control_message_conf))
            except Exception as e:
                logger.error("\nError converting payload to ControlMessage : %s", e)

        if (not self._disable_commit):
            consumer.commit(message=msg, asynchronous=self._async_commits)

        if self._stop_after > 0 and self._records_emitted >= self._stop_after:
            self.request_stop()

        return control_messages

    def _source_generator(self, subscription: mrc.Subscription):
        consumer = None
        try:
            consumer = ck.Consumer(self._consumer_params)
            consumer.subscribe(self._topics)

            do_sleep = False

            while not self.is_stop_requested() and subscription.is_subscribed():

                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    do_sleep = True

                else:
                    msg_error = msg.error()
                    if msg_error is None:
                        control_messages = self._process_msg(consumer, msg)
                        for control_message in control_messages:
                            yield control_message

                    elif msg_error == ck.KafkaError._PARTITION_EOF:
                        do_sleep = True
                    else:
                        raise ck.KafkaException(msg_error)

                if do_sleep and not self.is_stop_requested() and subscription.is_subscribed():
                    time.sleep(self._poll_interval)

        finally:
            # Close the consumer and call on_completed
            if (consumer):
                consumer.close()

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:
        return builder.make_source(self.unique_name, self._source_generator)
