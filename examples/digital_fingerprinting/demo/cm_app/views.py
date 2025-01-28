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

import logging

from cm_app.helper import KafkaWriter
from cm_app.helper import generate_success_message
from cm_app.helper import process_cm
from confluent_kafka import Producer
from flask import render_template
from flask import request

from . import app

KAFKA_WRITER = None

# pylint: disable=inconsistent-return-statements,no-member


@app.before_first_request
def setup():
    app.logger.setLevel(logging.INFO)
    producer = Producer({'bootstrap.servers': 'localhost:9092'})
    app.logger.info("Initialized Kafka producer")
    # pylint: disable=global-statement
    global KAFKA_WRITER
    KAFKA_WRITER = KafkaWriter(kafka_topic="test_cm", batch_size=1, producer=producer)
    app.logger.info("Initialized Kafka writer")


@app.route('/', methods=["GET", "POST"])
def submit_messages():

    if request.method == "POST":
        control_messages_json = process_cm(request)
        KAFKA_WRITER.write_data(control_messages_json)
        sucess_message = generate_success_message(control_messages_json)
        return sucess_message

    if request.method == "GET":
        return render_template("submit_messages.html")


@app.route('/training', methods=["GET", "POST"])
def training():

    if request.method == "POST":
        control_messages_json = process_cm(request)
        KAFKA_WRITER.write_data(control_messages_json)
        sucess_message = generate_success_message(control_messages_json)
        return sucess_message

    if request.method == "GET":
        return render_template("training.html")


@app.route('/review/results', methods=["GET"])
def reviewresults():

    if request.method == "GET":
        return render_template("review/results.html")
