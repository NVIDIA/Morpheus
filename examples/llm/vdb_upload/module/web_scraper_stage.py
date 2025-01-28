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

import logging
import typing

import mrc
from web_scraper_module import WebScraperLoaderFactory

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema

logger = logging.getLogger(f"morpheus.{__name__}")


class WebScraperStage(SinglePortStage):
    """
    Stage for scraping web based content using the HTTP GET protocol.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance.
    chunk_size : int
        Size in which to split the scraped content.
    link_column : str, default="link"
        Column which contains the links to scrape.
    enable_cache : bool, default = False
        Enables caching for requests data.
    cache_path : str, default="./.cache/http/RSSDownloadStage.sqlite"
        The path for the response caching system's sqlite database.
    """

    def __init__(self,
                 c: Config,
                 *,
                 chunk_size: int,
                 link_column: str = "link",
                 enable_cache: bool = False,
                 cache_path: str = "./.cache/http/RSSDownloadStage.sqlite"):
        super().__init__(c)

        self._module_config = {
            "web_scraper_config": {
                "link_column": link_column,
                "chunk_size": chunk_size,
                "enable_cache": enable_cache,
                "cache_path": cache_path,
                "cache_dir": "./.cache/llm/rss",
            }
        }

        self._input_port_name = "input"
        self._output_port_name = "output"

        self._module_loader = WebScraperLoaderFactory.get_instance("web_scraper", self._module_config)

    @property
    def name(self) -> str:
        """Returns the name of this stage."""
        return "rss-download"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple(`morpheus.pipeline.messages.MessageMeta`, )
            Accepted input types.

        """
        return (MessageMeta, )

    def supports_cpp_node(self):
        """Indicates whether this stage supports a C++ node."""
        return False

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(MessageMeta)

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        module = self._module_loader.load(builder=builder)

        mod_in_node = module.input_port(self._input_port_name)
        mod_out_node = module.output_port(self._output_port_name)

        builder.make_edge(input_node, mod_in_node)

        return mod_out_node
