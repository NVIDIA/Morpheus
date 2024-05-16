# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.http_client_source_stage import HttpClientSourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage


@pytest.mark.slow
@pytest.mark.use_cudf
@pytest.mark.parametrize("lines", [False, True], ids=["json", "lines"])
@pytest.mark.parametrize("use_payload_to_df_fn", [False, True], ids=["no_payload_to_df_fn", "payload_to_df_fn"])
def test_http_client_source_stage_pipe(config: Config,
                                       dataset: DatasetManager,
                                       mock_rest_server: str,
                                       lines: bool,
                                       use_payload_to_df_fn: bool):
    """
    Test the HttpClientSourceStage against a mock REST server which will return JSON data which can be deserialized
    into a DataFrame.
    """
    expected_df = dataset['filter_probs.csv']

    if lines:
        endpoint = "data-lines"
    else:
        endpoint = "data"

    if use_payload_to_df_fn:
        payload_to_df_fn = mock.MagicMock(return_value=expected_df[['v2', 'v3']].copy(deep=True))

        expected_df = expected_df[['v2', 'v3']].copy(deep=True)

        if lines:
            payload_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.jsonlines")
        else:
            payload_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.json")

        with open(payload_file, "rb") as f:
            expected_payload = f.read()

    else:
        payload_to_df_fn = None
        expected_payload = None

    url = f"{mock_rest_server}/api/v1/{endpoint}"

    num_records = len(expected_df)

    pipe = LinearPipeline(config)
    pipe.set_source(
        HttpClientSourceStage(config=config,
                              url=url,
                              max_retries=1,
                              lines=lines,
                              stop_after=num_records,
                              payload_to_df_fn=payload_to_df_fn))
    comp_stage = pipe.add_stage(CompareDataFrameStage(config, expected_df))
    pipe.run()

    assert_results(comp_stage.get_results())

    if use_payload_to_df_fn:
        payload_to_df_fn.assert_called_once_with(expected_payload, lines)
