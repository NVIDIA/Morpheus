# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import logging

logger = logging.getLogger(__name__)


class KafkaWriter:

    def __init__(self, kafka_topic, batch_size, producer):
        self._kafka_topic = kafka_topic
        self._batch_size = batch_size
        self._producer = producer

    @property
    def producer(self):
        return self._producer

    def write_data(self, message):
        self.producer.produce(self._kafka_topic, message.encode('utf-8'))
        if len(self.producer) >= self._batch_size:
            logger.info(
                "Batch reached, calling poll... producer unsent: %s",
                len(self.producer),
            )
            self.producer.flush()

    def close(self):
        logger.info("Closing kafka writer...")
        if self.producer is not None:
            self.producer.flush()
        logger.info("Closing kafka writer...Done")


def process_cm(request):
    control_messages_json = request.form.get("control-messages-json")

    logging.info("Received control message: %s", control_messages_json)

    return control_messages_json


def generate_success_message(control_messages_json):
    sucess_message = {
        "status": "Successfully published control message to kafka topic.",
        "status_code": 200,
        "control_messages": json.loads(control_messages_json)
    }

    sucess_message = json.dumps(sucess_message, indent=4)
    return sucess_message
