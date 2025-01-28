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
"""Utilities for testing Morpheus"""

import collections
import json
import os
import sys
import time
import types
import typing
from unittest import mock

import numpy as np
import pytest

from morpheus.io.deserializers import read_file_to_df

from .test_directories import TestDirectories

TEST_DIRS = TestDirectories(os.path.dirname(os.path.dirname(__file__)))

Results = collections.namedtuple('Results', ['total_rows', 'diff_rows', 'error_pct'])


def calc_error_val(results_file):
    """Based on the `calc_error_val` function in `val-utils.sh`."""
    with open(results_file, encoding='UTF-8') as fh:
        results = json.load(fh)

    total_rows = results['total_rows']
    diff_rows = results['diff_rows']
    return Results(total_rows=total_rows, diff_rows=diff_rows, error_pct=(diff_rows / total_rows) * 100)


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


# pylint: disable=inconsistent-return-statements
def require_env_variable(varname: str, reason: str, fail_missing: bool = False) -> str:
    """
    Checks if the given environment variable is set, and returns its value if it is. If the variable is not set, and
    `fail_missing` is False the test will ve skipped, otherwise a `RuntimeError` will be raised.
    """
    try:
        return os.environ[varname]
    except KeyError as e:
        if fail_missing:
            raise RuntimeError(reason) from e

        pytest.skip(reason=reason)


# pylint: enable=inconsistent-return-statements


def make_url(port: int, endpoint: str) -> str:
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint

    return f"http://127.0.0.1:{port}{endpoint}"


def make_mock_response(mock_request_session: mock.MagicMock,
                       status_code: int = 200,
                       content_type: str = "text/plain",
                       text: str = "test") -> mock.MagicMock:
    """
    Given a mocked `requests.Session` object, returns a mocked `requests.Response` object with the given status code
    """
    mock_response = mock.MagicMock()
    mock_response.status_code = status_code
    mock_response.headers = {"Content-Type": content_type}
    mock_response.text = text

    mock_request_session.return_value = mock_request_session
    mock_request_session.request.return_value = mock_response
    return mock_response


def mk_async_infer(inf_results: np.ndarray) -> typing.Callable:
    mock_infer_result = mock.MagicMock()
    mock_infer_result.as_numpy.side_effect = inf_results

    def async_infer(callback=None, **_):
        callback(mock_infer_result, None)

    return async_infer


def remove_module(mod_to_remove: str):
    """
    Remove a module, and all sub-modules from `sys.modules`. This is needed when testing examples which may import
    modules with common names such as `stages` which need to be removed from `sys.modules` before running tests for
    another example which might also contain its own `stages` module.
    """
    mod_prefix = f"{mod_to_remove}."
    for mod_name in list(sys.modules.keys()):
        if mod_name == mod_to_remove or mod_name.startswith(mod_prefix):
            del sys.modules[mod_name]


def load_json_file(filename):
    filepath = os.path.join(TEST_DIRS.tests_data_dir, filename)

    with open(filepath, 'r', encoding="utf-8") as json_file:
        return json.load(json_file)
