# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Utilities for testing Morpheus"""

import collections
import json
import os
import time
import types
import typing

import pytest

from morpheus.io.deserializers import read_file_to_df

from .test_directories import TestDirectories

TEST_DIRS = TestDirectories()

Results = collections.namedtuple('Results', ['total_rows', 'diff_rows', 'error_pct'])


def calc_error_val(results_file):
    """Based on the `calc_error_val` function in `val-utils.sh`."""
    with open(results_file, encoding='UTF-8') as fh:
        results = json.load(fh)

    total_rows = results['total_rows']
    diff_rows = results['diff_rows']
    return Results(total_rows=total_rows, diff_rows=diff_rows, error_pct=(diff_rows / total_rows) * 100)


def write_data_to_kafka(bootstrap_servers: str,
                        kafka_topic: str,
                        data: typing.List[typing.Union[str, dict]],
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

    assert num_records > 0
    time.sleep(1)

    return num_records


def write_file_to_kafka(bootstrap_servers: str,
                        kafka_topic: str,
                        input_file: str,
                        client_id: str = 'morpheus_unittest_writer') -> int:
    """
    Writes data from `inpute_file` into a given Kafka topic, emitting one message for each line int he file.
    Returning the number of messages written
    """
    with open(input_file, encoding='UTF-8') as fh:
        data = [line.strip() for line in fh]

    return write_data_to_kafka(bootstrap_servers=bootstrap_servers,
                               kafka_topic=kafka_topic,
                               data=data,
                               client_id=client_id)


def compare_class_to_scores(file_name, field_names, class_prefix, score_prefix, threshold):
    """
    Checks for expected columns in the dataframe which should be added by the `AddClassificationsStage` and
    `AddScoresStage` stages, and ensuring that the values produced by the `AddClassificationsStage` are consistent
    with the values produced by the `AddScoresStage` when the threshold is applied.
    """
    df = read_file_to_df(file_name, df_type='pandas')
    for field_name in field_names:
        class_field = f"{class_prefix}{field_name}"
        score_field = f"{score_prefix}{field_name}"
        above_thresh = df[score_field] > threshold

        assert all(above_thresh == df[class_field]), f"Mismatch on {field_name}"


def assert_path_exists(filename: str, retry_count: int = 5, delay_ms: int = 500):
    """
    This should be used in place of `assert os.path.exists(filename)` inside of tests. This will automatically retry
    with a delay if the file is not immediately found. This removes the need for adding any `time.sleep()` inside of
    tests

    Parameters
    ----------
    filename : str
        The path to assert exists
    retry_count : int, optional
        Number of times to check for the file before failing, by default 5
    delay_ms : int, optional
        Milliseconds between trys, by default 500

    Returns
    -------
    Returns none but will throw an assertion error on failure.
    """
    # Quick exit if the file exists
    if (os.path.exists(filename)):
        return

    attempts = 1

    # Otherwise, delay and retry
    while (attempts <= retry_count):
        time.sleep(delay_ms / 1000.0)

        if (os.path.exists(filename)):
            return

        attempts += 1

    # Finally, actually assert on the final try
    assert os.path.exists(filename)


def assert_results(results: dict) -> dict:
    """
    Receives the results dict from the `CompareDataframeStage.get_results` method,
    and asserts that all columns and rows match
    """
    assert results["diff_cols"] == 0, f"Expected diff_cols=0 : {results}"
    assert results["diff_rows"] == 0, f"Expected diff_rows=0 : {results}"
    return results


def import_or_skip(modname: str,
                   minversion: str = None,
                   reason: str = None,
                   fail_missing: bool = False) -> types.ModuleType:
    """
    Wrapper for `pytest.importorskip` will re-raise any `Skipped` exceptions as `ImportError` if `fail_missing` is True.
    """
    try:
        return pytest.importorskip(modname, minversion=minversion, reason=reason)
    except pytest.skip.Exception as e:
        if fail_missing:
            raise ImportError(e) from e
        raise
