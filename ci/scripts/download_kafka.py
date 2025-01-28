#!/usr/bin/env python3
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
"""Download Kafka for testing."""

import os

import pytest_kafka
from pytest_kafka.install import set_up_kafka

DEFAULT_KAFKA_URL = 'https://downloads.apache.org/kafka/3.8.0/kafka_2.13-3.8.0.tgz'
DEFAULT_KAFKA_TAR_ROOTDIR = 'kafka_2.13-3.8.0/'


def main():
    """Main function."""
    kafka_url = os.environ.get('MORPHEUS_KAFKA_URL', DEFAULT_KAFKA_URL)
    kafka_tar_dir = os.environ.get('MORPHEUS_KAFKA_TAR_DIR', DEFAULT_KAFKA_TAR_ROOTDIR)
    pytest_kafka_dir = os.path.dirname(pytest_kafka.__file__)
    set_up_kafka(kafka_url=kafka_url, kafka_tar_rootdir=kafka_tar_dir, extract_location=pytest_kafka_dir)


if __name__ == '__main__':
    main()
