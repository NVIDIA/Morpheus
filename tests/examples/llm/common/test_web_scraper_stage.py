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
import types

import pytest

import cudf

from _utils import TEST_DIRS
from _utils import assert_results
from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage


@pytest.mark.slow
@pytest.mark.import_mod(os.path.join(TEST_DIRS.examples_dir, 'llm/vdb_upload/module/web_scraper_stage.py'))
def test_http_client_source_stage_pipe(config: Config, mock_rest_server: str, import_mod: types.ModuleType):
    url = f"{mock_rest_server}/www/index"

    df = cudf.DataFrame({"link": [url]})

    df_expected = cudf.DataFrame({"link": [url], "page_content": "website title some paragraph"})

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [df]))
    pipe.add_stage(import_mod.WebScraperStage(config, chunk_size=config.feature_length))
    comp_stage = pipe.add_stage(CompareDataFrameStage(config, compare_df=df_expected))

    pipe.run()

    assert_results(comp_stage.get_results())
