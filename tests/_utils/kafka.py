# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import subprocess
import time
import typing
import warnings
from collections import namedtuple
from functools import partial

import pytest

if (typing.TYPE_CHECKING):
    from kafka import KafkaConsumer

# Amount of seconds we will wait for the Kafka consumer to be assigned partitions
PARTITION_ASSIGNMENT_TIMEOUT = 15

# actual topic names not important, but we will need two of them.
KafkaTopics = namedtuple('KafkaTopics', ['input_topic', 'output_topic'])

# pylint: disable=invalid-name
_kafka_consumer = None
kafka_server = None
zookeeper_proc = None
# pylint: enable=invalid-name


@pytest.fixture(name='kafka_topics', scope='function')
def kafka_topics_fixture():
    """
    Every test receives a unique pair of Kafka topics
    """
    ts = time.time()
    yield KafkaTopics(f'morpheus_unittest_input_{ts}', f'morpheus_unittest_output_{ts}')


@pytest.fixture(name='kafka_bootstrap_servers', scope="function")
def kafka_bootstrap_servers_fixture(kafka_server: tuple[subprocess.Popen, int]):  # pylint: disable=redefined-outer-name
    """
    Used by tests that require both an input and an output topic
    """
    kafka_port = kafka_server[1]
    yield f"localhost:{kafka_port}"


def seek_to_beginning(kafka_consumer: "KafkaConsumer", timeout: int = PARTITION_ASSIGNMENT_TIMEOUT):
    """
    Seeks to the beginning of the Kafka topic
    """
    start = time.time()
    end = start + timeout
    partitions_assigned = False
    while not partitions_assigned and time.time() <= end:
        kafka_consumer.poll(timeout_ms=20)
        partitions_assigned = len(kafka_consumer.assignment()) > 0
        if not partitions_assigned:
            time.sleep(0.1)

    assert partitions_assigned

    kafka_consumer.seek_to_beginning()


@pytest.fixture(name='kafka_consumer', scope='function')
def kafka_consumer_fixture(kafka_topics: KafkaTopics, _kafka_consumer: "KafkaConsumer"):
    with warnings.catch_warnings():
        # Ignore warnings specific to the test fixture and not the actual morpheus code
        warnings.filterwarnings("ignore",
                                message=r"Exception ignored in:.*ConsumerCoordinator\.__del__",
                                category=pytest.PytestUnraisableExceptionWarning)
        _kafka_consumer.subscribe([kafka_topics.output_topic])
        seek_to_beginning(_kafka_consumer)

        yield _kafka_consumer


def _init_pytest_kafka() -> (bool, Exception):
    """
    Since the Kafka tests don't run by default, we will silently fail to initialize unless --run_kafka is enabled.
    This should only be called once by the root conftest.py
    """
    global kafka_server, _kafka_consumer, zookeeper_proc  # pylint: disable=global-statement
    try:
        import pytest_kafka
        os.environ['KAFKA_OPTS'] = "-Djava.net.preferIPv4Stack=True"
        # Initialize pytest_kafka fixtures following the recomendations in:
        # https://gitlab.com/karolinepauls/pytest-kafka/-/blob/master/README.rst
        kafka_scripts = os.path.join(os.path.dirname(pytest_kafka.__file__), 'kafka/bin/')
        if not os.path.exists(kafka_scripts):
            # check the old location
            kafka_scripts = os.path.join(os.path.dirname(os.path.dirname(pytest_kafka.__file__)), 'kafka/bin/')

        kafka_bin = os.path.join(kafka_scripts, 'kafka-server-start.sh')
        zookeeper_bin = os.path.join(kafka_scripts, 'zookeeper-server-start.sh')

        for kafka_script in (kafka_bin, zookeeper_bin):
            if not os.path.exists(kafka_script):
                raise RuntimeError(f"Required Kafka script not found: {kafka_script}")

        teardown_fn = partial(pytest_kafka.terminate, signal_fn=subprocess.Popen.kill)
        zookeeper_proc = pytest_kafka.make_zookeeper_process(zookeeper_bin, teardown_fn=teardown_fn, scope='session')
        kafka_server = pytest_kafka.make_kafka_server(kafka_bin,
                                                      'zookeeper_proc',
                                                      teardown_fn=teardown_fn,
                                                      scope='session')
        _kafka_consumer = pytest_kafka.make_kafka_consumer('kafka_server',
                                                           scope='function',
                                                           group_id='morpheus_unittest_consumer')

        return (True, None)
    except Exception as e:
        return (False, e)


def write_data_to_kafka(bootstrap_servers: str,
                        kafka_topic: str,
                        data: list[str | dict],
                        client_id: str = 'morpheus_unittest_writer') -> int:
    """
    Writes `data` into a given Kafka topic, emitting one message for each line int he file. Returning the number of
    messages written
    """
    # pylint: disable=import-error
    from kafka import KafkaProducer
    num_records = 0
    producer = KafkaProducer(bootstrap_servers=bootstrap_servers, client_id=client_id)
    for row in data:
        if isinstance(row, dict):
            row = json.dumps(row)
        producer.send(kafka_topic, row.encode('utf-8'))
        num_records += 1

    producer.flush()

    assert num_records == len(data)

    return num_records


def write_file_to_kafka(bootstrap_servers: str,
                        kafka_topic: str,
                        input_file: str,
                        client_id: str = 'morpheus_unittest_writer') -> int:
    """
    Writes data from `input_file` into a given Kafka topic, emitting one message for each line in the file, and returns
    the number of messages written.
    """
    with open(input_file, encoding='UTF-8') as fh:
        data = [line.strip() for line in fh]

    return write_data_to_kafka(bootstrap_servers=bootstrap_servers,
                               kafka_topic=kafka_topic,
                               data=data,
                               client_id=client_id)
