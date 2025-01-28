# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
import typing
from enum import Enum
from io import StringIO

import confluent_kafka as ck
import mrc
import pandas as pd

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.config import auto_determine_bootstrap
from morpheus.io.utils import get_json_reader
from morpheus.messages import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.utils.type_aliases import DataFrameType

logger = logging.getLogger(__name__)


class AutoOffsetReset(Enum):
    """The supported offset options in Kafka"""
    EARLIEST = "earliest"
    LATEST = "latest"
    NONE = "none"


@register_stage("from-kafka", modes=[PipelineModes.FIL, PipelineModes.NLP, PipelineModes.OTHER])
class KafkaSourceStage(PreallocatorMixin, GpuAndCpuMixin, SingleOutputSource):
    """
    Load messages from a Kafka cluster.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    bootstrap_servers : str
        Comma-separated list of bootstrap servers. If using Kafka created via `docker-compose`, this can be set to
        'auto' to automatically determine the cluster IPs and ports
    input_topic : typing.List[str], default = ["test_pcap"]
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
                 config: Config,
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
        super().__init__(config)

        if (input_topic is None):
            input_topic = ["test_pcap"]

        if isinstance(auto_offset_reset, AutoOffsetReset):
            auto_offset_reset = auto_offset_reset.value

        if (bootstrap_servers == "auto"):
            bootstrap_servers = auto_determine_bootstrap()

        self._consumer_params = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'session.timeout.ms': "60000",
            "auto.offset.reset": auto_offset_reset
        }
        if client_id is not None:
            self._consumer_params['client.id'] = client_id

        if isinstance(input_topic, str):
            input_topic = [input_topic]

        # Remove duplicate topics if there are any.
        self._topics = list(set(input_topic))
        self._max_batch_size = config.pipeline_batch_size
        self._max_concurrent = config.num_threads
        self._disable_commit = disable_commit
        self._disable_pre_filtering = disable_pre_filtering
        self._stop_after = stop_after
        self._async_commits = async_commits
        self._client = None

        self._poll_interval = pd.Timedelta(poll_interval).total_seconds()
        self._started = False

        # Defined lated if in CPU mode
        self._json_reader: typing.Callable[..., DataFrameType] = None

        self._records_emitted = 0
        self._num_messages = 0

    @property
    def name(self) -> str:
        return "from-kafka"

    def supports_cpp_node(self):
        return True

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(MessageMeta)

    def _process_batch(self, consumer, batch):
        message_meta = None
        if len(batch):
            buffer = StringIO()

            for msg in batch:
                payload = msg.value()
                if payload is not None:
                    buffer.write(payload.decode("utf-8"))
                    buffer.write("\n")

            df = None
            try:
                buffer.seek(0)
                df = self._json_reader(buffer, lines=True, orient='records')
            except Exception as e:
                logger.error("Error parsing payload into a dataframe : %s", e)
            finally:
                if (not self._disable_commit):
                    for msg in batch:
                        consumer.commit(message=msg, asynchronous=self._async_commits)

            if df is not None:
                num_records = len(df)
                message_meta = MessageMeta(df)
                self._records_emitted += num_records
                self._num_messages += 1

                if self._stop_after > 0 and self._records_emitted >= self._stop_after:
                    self.request_stop()

            batch.clear()

        return message_meta

    def _source_generator(self, subscription: mrc.Subscription):
        consumer = None
        try:
            consumer = ck.Consumer(self._consumer_params)
            consumer.subscribe(self._topics)

            batch = []

            while not self.is_stop_requested() and subscription.is_subscribed():
                do_process_batch = False
                do_sleep = False

                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    do_process_batch = True
                    do_sleep = True

                else:
                    msg_error = msg.error()
                    if msg_error is None:
                        batch.append(msg)
                        if len(batch) == self._max_batch_size:
                            do_process_batch = True

                    elif msg_error == ck.KafkaError._PARTITION_EOF:
                        do_process_batch = True
                        do_sleep = True
                    else:
                        raise ck.KafkaException(msg_error)

                if do_process_batch:
                    message_meta = self._process_batch(consumer, batch)
                    if message_meta is not None:
                        yield message_meta

                if do_sleep and not self.is_stop_requested():
                    time.sleep(self._poll_interval)

            message_meta = self._process_batch(consumer, batch)
            if message_meta is not None:
                yield message_meta

        finally:
            # Close the consumer and call on_completed
            if (consumer):
                consumer.close()

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:

        if (self._build_cpp_node()):
            import morpheus._lib.stages as _stages
            source = _stages.KafkaSourceStage(builder,
                                              self.unique_name,
                                              self._max_batch_size,
                                              self._topics,
                                              int(self._poll_interval * 1000),
                                              self._consumer_params,
                                              self._disable_commit,
                                              self._disable_pre_filtering,
                                              self._stop_after,
                                              self._async_commits)

            # Only use multiple progress engines with C++. The python implementation will duplicate messages with
            # multiple threads
            source.launch_options.pe_count = self._max_concurrent
        else:
            self._json_reader = get_json_reader(self._config.execution_mode)
            source = builder.make_source(self.unique_name, self._source_generator)

        return source
