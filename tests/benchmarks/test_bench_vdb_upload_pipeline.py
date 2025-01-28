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
"""
Benchmark the examples/llm/vdb_upload/pipeline.py example, talking with live Triton and Milvus servers, but mocked
RSS and web scraper responses."""

import collections.abc
import json
import os
import time
import types
import typing
from unittest import mock

import pytest

from _utils import TEST_DIRS
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.input.rss_source_stage import RSSSourceStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage
from morpheus_llm.stages.output.write_to_vector_db_stage import WriteToVectorDBStage

EMBEDDING_SIZE = 384
MODEL_MAX_BATCH_SIZE = 64
MODEL_FEA_LENGTH = 512


@pytest.fixture(name="pymilvus", scope='session', autouse=True)
def pymilvus_fixture(pymilvus: types.ModuleType):
    """
    Fixture to ensure pymilvus is installed
    """
    yield pymilvus


def _run_pipeline(config: Config,
                  milvus_server_uri: str,
                  collection_name: str,
                  rss_urls: list[str],
                  utils_mod: types.ModuleType,
                  web_scraper_stage_mod: types.ModuleType):

    config.mode = PipelineModes.NLP
    config.pipeline_batch_size = 1024
    config.model_max_batch_size = MODEL_MAX_BATCH_SIZE
    config.feature_length = MODEL_FEA_LENGTH
    config.edge_buffer_size = 128
    config.class_labels = [str(i) for i in range(EMBEDDING_SIZE)]

    pipe = LinearPipeline(config)

    pipe.set_source(
        RSSSourceStage(config, feed_input=rss_urls, batch_size=128, run_indefinitely=False, enable_cache=False))
    pipe.add_stage(web_scraper_stage_mod.WebScraperStage(config, chunk_size=MODEL_FEA_LENGTH, enable_cache=False))
    pipe.add_stage(DeserializeStage(config))

    pipe.add_stage(
        PreprocessNLPStage(config,
                           vocab_hash_file=os.path.join(TEST_DIRS.data_dir, 'bert-base-uncased-hash.txt'),
                           do_lower_case=True,
                           truncation=True,
                           add_special_tokens=False,
                           column='page_content'))

    pipe.add_stage(
        TritonInferenceStage(config,
                             model_name='all-MiniLM-L6-v2',
                             server_url='localhost:8001',
                             force_convert_inputs=True))

    pipe.add_stage(
        WriteToVectorDBStage(config,
                             resource_name=collection_name,
                             resource_kwargs=utils_mod.build_default_milvus_config(embedding_size=EMBEDDING_SIZE),
                             recreate=True,
                             service="milvus",
                             uri=milvus_server_uri))
    pipe.run()


@pytest.mark.milvus
@pytest.mark.cpu_mode
@pytest.mark.use_pandas
@pytest.mark.benchmark
@pytest.mark.import_mod([
    os.path.join(TEST_DIRS.examples_dir, 'llm/common/utils.py'),
    os.path.join(TEST_DIRS.examples_dir, 'llm/vdb_upload/module/web_scraper_stage.py'),
])
@mock.patch('feedparser.http.get')
@mock.patch('requests.Session')
def test_vdb_upload_pipe(mock_requests_session: mock.MagicMock,
                         mock_feedparser_http_get: mock.MagicMock,
                         mock_web_scraper_request_time: float,
                         mock_feedparser_request_time: float,
                         benchmark: collections.abc.Callable[[collections.abc.Callable], typing.Any],
                         config: Config,
                         milvus_server_uri: str,
                         import_mod: list[types.ModuleType]):

    with open(os.path.join(TEST_DIRS.tests_data_dir, 'service/cisa_web_responses.json'), encoding='utf-8') as fh:
        web_responses = json.load(fh)

    def mock_get_fn(url: str):
        mock_response = mock.MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.text = web_responses[url]
        time.sleep(mock_web_scraper_request_time)
        return mock_response

    mock_requests_session.return_value = mock_requests_session
    mock_requests_session.get.side_effect = mock_get_fn

    def mock_feedparser_http_get_fn(*args, **kwargs):  # pylint: disable=unused-argument
        time.sleep(mock_feedparser_request_time)
        # The RSS Parser expects a bytes string
        with open(os.path.join(TEST_DIRS.tests_data_dir, 'service/cisa_rss_feed.xml'), 'rb') as fh:
            return fh.read()

    mock_feedparser_http_get.side_effect = mock_feedparser_http_get_fn

    (utils_mod, web_scraper_stage_mod) = import_mod
    collection_name = "test_bench_vdb_upload_pipeline"

    benchmark(_run_pipeline,
              config=config,
              milvus_server_uri=milvus_server_uri,
              collection_name=collection_name,
              rss_urls=["https://www.us-cert.gov/ncas/current-activity.xml"],
              utils_mod=utils_mod,
              web_scraper_stage_mod=web_scraper_stage_mod)
