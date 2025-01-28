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
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage


@pytest.mark.slow
@pytest.mark.import_mod(os.path.join(TEST_DIRS.examples_dir, 'llm/vdb_upload/module/web_scraper_module.py'))
def test_web_scraper_module(config: Config, mock_rest_server: str, import_mod: types.ModuleType):
    url = f"{mock_rest_server}/www/index"

    df = cudf.DataFrame({"link": [url]})
    df_expected = cudf.DataFrame({"link": [url], "page_content": "website title some paragraph"})

    web_scraper_loader = import_mod.WebScraperLoaderFactory.get_instance(
        "web_scraper",
        module_config={
            "web_scraper_config": {
                "link_column": "link",
                "chunk_size": 100,
                "enable_cache": False,
                "cache_path": "./.cache/http/RSSDownloadStage.sqlite",
                "cache_dir": "./.cache/llm/rss"
            }
        })

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [df]))
    pipe.add_stage(
        LinearModulesStage(config,
                           web_scraper_loader,
                           input_type=MessageMeta,
                           output_type=MessageMeta,
                           input_port_name="input",
                           output_port_name="output"))
    comp_stage = pipe.add_stage(CompareDataFrameStage(config, compare_df=df_expected))
    pipe.run()

    assert_results(comp_stage.get_results())
