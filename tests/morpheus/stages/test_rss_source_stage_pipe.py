#!/usr/bin/env python
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

import os

import pytest

from _utils import TEST_DIRS
from morpheus.config import Config
from morpheus.pipeline.pipeline import Pipeline
from morpheus.stages.input.rss_source_stage import RSSSourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage

valid_feed_input = os.path.join(TEST_DIRS.tests_data_dir, "rss_feed_atom.xml")
invalid_feed_input = os.path.join(TEST_DIRS.tests_data_dir, "rss_feed_atom.xm")


def test_support_cpp_node(config):
    url_feed_input = "https://fake.nvidia.com/rss/HomePage.xml"
    rss_source_stage = RSSSourceStage(config, feed_input=[url_feed_input])

    assert rss_source_stage.supports_cpp_node() is False


@pytest.mark.parametrize(
    "feed_input, batch_size, expected_count, enable_cache",
    [([valid_feed_input], 30, 1, False), ([valid_feed_input], 12, 3, True),
     ([valid_feed_input, valid_feed_input], 15, 2, False)
     # Duplicate feed inputs
     ])
def test_rss_source_stage_pipe(config: Config,
                               feed_input: list[str],
                               batch_size: int,
                               expected_count: int,
                               enable_cache: bool):
    pipe = Pipeline(config)

    rss_source_stage = pipe.add_stage(
        RSSSourceStage(config, feed_input=feed_input, batch_size=batch_size, enable_cache=enable_cache))
    sink_stage = pipe.add_stage(InMemorySinkStage(config))

    pipe.add_edge(rss_source_stage, sink_stage)

    pipe.run()

    assert len(sink_stage.get_messages()) == expected_count


def test_invalid_input_rss_source_stage(config: Config):

    with pytest.raises(ValueError, match=f"Invalid URL or file path: {invalid_feed_input}"):
        RSSSourceStage(config, feed_input=[invalid_feed_input], interval_secs=1, cooldown_interval=500)
