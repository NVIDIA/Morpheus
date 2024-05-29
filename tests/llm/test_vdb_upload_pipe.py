# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Mimic the examples/llm/vdb_upload/pipeline.py example"""

import json
import os
import types
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from _utils import TEST_DIRS
from _utils import mk_async_infer
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.service.vdb.milvus_vector_db_service import MilvusVectorDBServiceProvider


@pytest.mark.milvus
@pytest.mark.use_python
@pytest.mark.use_pandas
@pytest.mark.import_mod([
    os.path.join(TEST_DIRS.examples_dir, 'llm/common'),
    os.path.join(TEST_DIRS.examples_dir, 'llm/vdb_upload/helper.py'),
    os.path.join(TEST_DIRS.examples_dir, 'llm/vdb_upload/run.py'),
    os.path.join(TEST_DIRS.examples_dir, 'llm/vdb_upload/pipeline.py')
])
@mock.patch('requests.Session')
@mock.patch('tritonclient.grpc.InferenceServerClient')
@pytest.mark.parametrize('is_rss_source, exclude_columns, expected_output_path, vdb_conf_file',
                         [(True, ['id', 'embedding', 'source'],
                           'service/milvus_rss_data.json',
                           'examples/llm/vdb_upload/vdb_rss_source_config.yaml'),
                          (False, ['id', 'embedding'],
                           'examples/llm/vdb_upload/test_data_output.json',
                           'examples/llm/vdb_upload/vdb_file_source_config.yaml')])
@pytest.mark.skip(reason="Test is broken because of a bad merge. Re-enable once config.yamls are fixed.")
def test_vdb_upload_pipe(mock_triton_client: mock.MagicMock,
                         mock_requests_session: mock.MagicMock,
                         dataset: DatasetManager,
                         milvus_server_uri: str,
                         import_mod: list[types.ModuleType],
                         is_rss_source: str,
                         exclude_columns: list[str],
                         expected_output_path: str,
                         vdb_conf_file: str):

    # We're going to use this DF to both provide values to the mocked Tritonclient,
    # but also to verify the values in the Milvus collection.
    expected_values_df = dataset[expected_output_path]

    if is_rss_source:
        with open(os.path.join(TEST_DIRS.tests_data_dir, 'service/cisa_web_responses.json'), encoding='utf-8') as fh:
            web_responses = json.load(fh)

        # Mock requests, since we are feeding the RSSSourceStage with a local file it won't be using the
        # requests lib, only web_scraper_stage.py will use it.
        def mock_get_fn(url: str):
            mock_response = mock.MagicMock()
            mock_response.ok = True
            mock_response.status_code = 200
            mock_response.text = web_responses[url]
            return mock_response

        mock_requests_session.return_value = mock_requests_session
        mock_requests_session.get.side_effect = mock_get_fn

        # As page_content is used by other pipelines, we're just renaming it to content.
        expected_values_df = expected_values_df.rename(columns={"page_content": "content"})
        expected_values_df["source"] = "rss"

    vdb_conf_path = os.path.join(TEST_DIRS.tests_data_dir, vdb_conf_file)

    _, _, vdb_upload_run_mod, vdb_upload_pipeline_mod = import_mod

    # Building final configuration. Here we're passing empty dictionaries for cli configuration.
    vdb_pipeline_config = vdb_upload_run_mod.build_final_config(vdb_conf_path=vdb_conf_path,
                                                                cli_source_conf={},
                                                                cli_embeddings_conf={},
                                                                cli_pipeline_conf={},
                                                                cli_tokenizer_conf={},
                                                                cli_vdb_conf={})

    config: Config = vdb_pipeline_config["pipeline_config"]

    # Overwriting uri provided in the config file with milvus_server_uri
    vdb_pipeline_config["vdb_config"]["uri"] = milvus_server_uri
    collection_name = vdb_pipeline_config["vdb_config"]["resource_name"]

    # Mock Triton results
    mock_metadata = {
        "inputs": [{
            "name": "input_ids", "datatype": "INT32", "shape": [-1, config.feature_length]
        }, {
            "name": "attention_mask", "datatype": "INT32", "shape": [-1, config.feature_length]
        }],
        "outputs": [{
            "name": "output", "datatype": "FP32", "shape": [-1, len(config.class_labels)]
        }]
    }
    mock_model_config = {"config": {"max_batch_size": 256}}

    mock_triton_client.return_value = mock_triton_client
    mock_triton_client.is_server_live.return_value = True
    mock_triton_client.is_server_ready.return_value = True
    mock_triton_client.is_model_ready.return_value = True
    mock_triton_client.get_model_metadata.return_value = mock_metadata
    mock_triton_client.get_model_config.return_value = mock_model_config

    mock_result_values = expected_values_df['embedding'].to_list()
    inf_results = np.split(mock_result_values,
                           range(config.model_max_batch_size, len(mock_result_values), config.model_max_batch_size))

    # The triton client is going to perform a logits function, calculate the inverse of it here
    inf_results = [np.log((1.0 / x) - 1.0) * -1 for x in inf_results]

    async_infer = mk_async_infer(inf_results)
    mock_triton_client.async_infer.side_effect = async_infer

    vdb_upload_pipeline_mod.pipeline(**vdb_pipeline_config)

    milvus_provider = MilvusVectorDBServiceProvider(uri=milvus_server_uri)
    milvus_service = milvus_provider.create()
    resource_service = milvus_service.load_resource(name=collection_name)

    assert resource_service.count() == len(expected_values_df)

    db_results = resource_service.query("", offset=0, limit=resource_service.count())
    db_df = pd.DataFrame(sorted(db_results, key=lambda k: k['id']))

    # The comparison function performs rounding on the values, but is unable to do so for array columns
    dataset.assert_compare_df(db_df, expected_values_df[db_df.columns], exclude_columns=exclude_columns)
    db_emb = db_df['embedding']
    expected_emb = expected_values_df['embedding']

    for i in range(resource_service.count()):
        db_emb_row = pd.DataFrame(db_emb[i], dtype=np.float32)
        expected_emb_row = pd.DataFrame(expected_emb[i], dtype=np.float32)
        dataset.assert_compare_df(db_emb_row, expected_emb_row)

    milvus_service.close()
