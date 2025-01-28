# Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

import os
from unittest import mock

import pytest

from _utils import TEST_DIRS
from _utils import assert_results
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.configurable_output_source import SupportedMessageTypes
from morpheus.stages.input.http_client_source_stage import HttpClientSourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage


@pytest.mark.slow
@pytest.mark.use_cudf
@pytest.mark.parametrize("lines", [False, True], ids=["json", "lines"])
@pytest.mark.parametrize("use_payload_to_df_fn", [False, True], ids=["no_payload_to_df_fn", "payload_to_df_fn"])
@pytest.mark.parametrize("message_type, task_type, task_payload",
                         [(SupportedMessageTypes.MESSAGE_META, None, None),
                          (SupportedMessageTypes.CONTROL_MESSAGE, None, None),
                          (SupportedMessageTypes.CONTROL_MESSAGE, "test", {
                              "pay": "load"
                          })],
                         ids=["message_meta", "control_message_no_task", "control_message_with_task"])
def test_http_client_source_stage_pipe(config: Config,
                                       dataset: DatasetManager,
                                       mock_rest_server: str,
                                       lines: bool,
                                       use_payload_to_df_fn: bool,
                                       message_type: SupportedMessageTypes,
                                       task_type: str | None,
                                       task_payload: dict | None):
    """
    Test the HttpClientSourceStage against a mock REST server which will return JSON data which can be deserialized
    into a DataFrame.
    """
    source_df = dataset['filter_probs.csv']

    if lines:
        endpoint = "data-lines"
    else:
        endpoint = "data"

    if use_payload_to_df_fn:
        if lines:
            payload_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.jsonlines")
        else:
            payload_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.json")

        with open(payload_file, "rb") as f:
            expected_payload = f.read()

        def payload_to_df_fn(payload, lines_arg):
            assert payload == expected_payload
            assert lines_arg == lines
            return source_df[['v2', 'v3']].copy(deep=True)

        expected_df = source_df[['v2', 'v3']].copy(deep=True)

    else:
        payload_to_df_fn = None
        expected_payload = None
        expected_df = source_df.copy(deep=True)

    url = f"{mock_rest_server}/api/v1/{endpoint}"

    num_records = len(expected_df)

    pipe = LinearPipeline(config)
    pipe.set_source(
        HttpClientSourceStage(config=config,
                              url=url,
                              max_retries=1,
                              lines=lines,
                              stop_after=num_records,
                              payload_to_df_fn=payload_to_df_fn,
                              message_type=message_type,
                              task_type=task_type,
                              task_payload=task_payload))
    comp_stage = pipe.add_stage(CompareDataFrameStage(config, expected_df))
    pipe.run()

    assert_results(comp_stage.get_results(clear=False))

    messages = comp_stage.get_messages()
    assert len(messages) == 1

    recv_msg = messages[0]
    if message_type == SupportedMessageTypes.MESSAGE_META:
        assert isinstance(recv_msg, MessageMeta)
    else:
        assert isinstance(recv_msg, ControlMessage)
        if task_type is not None:
            expected_tasks = {task_type: [task_payload]}
        else:
            expected_tasks = {}

        assert recv_msg.get_tasks() == expected_tasks

        if lines:
            # This is the content type specified in `tests/mock_rest_server/mocks/api/v1/data-lines/GET.mock`
            expected_content_type = "text/plain;charset=UTF-8"
        else:
            expected_content_type = "application/json"

        # Subset of headers that we want to check for
        expected_headers = {'url': url, 'method': 'GET', 'Content-Type': expected_content_type}

        actual_headers = recv_msg.get_metadata()['http_fields']
        for (key, value) in expected_headers.items():
            assert actual_headers[key] == value


@pytest.mark.slow
@pytest.mark.use_cudf
@pytest.mark.parametrize(
    "lines",
    [False, pytest.param(True, marks=pytest.mark.skip(reason="https://github.com/rapidsai/cudf/issues/15820"))],
    ids=["json", "lines"])
@pytest.mark.parametrize("use_payload_to_df_fn", [False, True], ids=["no_payload_to_df_fn", "payload_to_df_fn"])
def test_parse_errors(config: Config, mock_rest_server: str, lines: bool, use_payload_to_df_fn: bool):
    url = f"{mock_rest_server}/api/v1/invalid"

    if use_payload_to_df_fn:
        payload_to_df_fn = mock.MagicMock(side_effect=ValueError("Invalid payload"))
    else:
        payload_to_df_fn = None

    pipe = LinearPipeline(config)
    pipe.set_source(
        HttpClientSourceStage(config=config, url=url, max_retries=1, lines=lines, payload_to_df_fn=payload_to_df_fn))

    # cudf raises a RuntimeError when it should be raising a ValueError also a part of #15820
    with pytest.raises(Exception):
        pipe.run()

    if use_payload_to_df_fn:
        payload_to_df_fn.assert_called_once()
