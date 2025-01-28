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

import time
from os import path
from unittest.mock import Mock
from unittest.mock import patch
from xml.etree import ElementTree

import feedparser
import pytest
from bs4 import BeautifulSoup

import cudf

from _utils import TEST_DIRS
from morpheus.controllers.rss_controller import FeedStats
from morpheus.controllers.rss_controller import RSSController
from morpheus.utils.type_aliases import SeriesType

test_urls = ["https://fake.nvidia.com/rss/HomePage.xml"]

test_invalid_urls = [
    "invalid_url",
    "example.com/rss-feed-url",
    "ftp://",
]

test_file_paths = [path.join(TEST_DIRS.tests_data_dir, "rss_feed_atom.xml")]

test_invalid_file_paths = [
    "/path/to/nonexistent_file.xml",
    "/path/to/directory/",
]


@pytest.fixture(scope="module", name="mock_feed")
def mock_feed_fixture() -> feedparser.FeedParserDict:
    feed_items = [{"link": "https://nvidia.com", "id": "12345"}, {"link": "https://fake.nvidia.com", "id": "22345"}]
    feed = feedparser.FeedParserDict()
    feed.update({"entries": feed_items, "bozo": 0})

    return feed


@pytest.fixture(scope="module", name="mock_get_response")
def mock_get_response_fixture() -> Mock:
    # Open and read the content of the file
    with open(test_file_paths[0], 'rb') as file:
        file_content = file.read()

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = file_content
    mock_response.text = file_content

    return mock_response


@pytest.fixture(scope="module", name="cisa_rss_feed")
def cisa_rss_feed_fixture() -> str:
    return [path.join(TEST_DIRS.tests_data_dir, 'service/cisa_rss_feed.xml')]


@pytest.mark.parametrize("feed_input, expected_output", [(url, True) for url in test_urls])
def test_run_indefinitely_true(feed_input: str, expected_output: bool):
    controller = RSSController(feed_input=feed_input)
    assert controller.run_indefinitely == expected_output


@pytest.mark.parametrize("feed_input", test_file_paths)
def test_run_indefinitely_false(feed_input: list[str]):
    controller = RSSController(feed_input=feed_input)
    assert controller.run_indefinitely is False


@pytest.mark.parametrize("feed_input", test_urls)
def test_parse_feed_valid_url(feed_input: list[str], mock_feed: feedparser.FeedParserDict, mock_get_response: Mock):
    controller = RSSController(feed_input=feed_input)

    patch("morpheus.controllers.rss_controller.feedparser.parse", return_value=mock_feed)

    with patch("requests.Session.get", return_value=mock_get_response):
        feed = list(controller.parse_feeds())[0]
        assert feed.entries


@pytest.mark.parametrize("feed_input", test_invalid_urls + test_invalid_file_paths)
def test_parse_feed_invalid_input(feed_input: list[str]):
    with pytest.raises(ValueError, match=f"Invalid URL or file path: {feed_input}"):
        RSSController(feed_input=feed_input)


@pytest.mark.parametrize("strip_markup", [False, True])
@pytest.mark.parametrize("feed_input, expected_count", [(test_file_paths[0], 30)])
def test_skip_duplicates_feed_inputs(feed_input: str, expected_count: int, strip_markup: bool):
    controller = RSSController(feed_input=[feed_input, feed_input],
                               strip_markup=strip_markup)  # Pass duplicate feed inputs
    dataframes_generator = controller.fetch_dataframes()
    dataframe = next(dataframes_generator, None)
    assert isinstance(dataframe, cudf.DataFrame)
    assert len(dataframe) == expected_count


@pytest.mark.parametrize("feed_input", test_urls)
def test_is_url_true(feed_input: list[str]):
    assert RSSController.is_url(feed_input)


@pytest.mark.parametrize("feed_input", test_invalid_urls + test_invalid_file_paths + test_file_paths)
def test_is_url_false(feed_input: list[str]):
    assert not RSSController.is_url(feed_input)


@pytest.mark.parametrize("feed_input", [test_urls, test_urls[0]])
def test_fetch_dataframes_url(feed_input: str | list[str],
                              mock_feed: feedparser.FeedParserDict,
                              mock_get_response: Mock):
    controller = RSSController(feed_input=feed_input)

    patch("morpheus.controllers.rss_controller.feedparser.parse", return_value=mock_feed)

    with patch("requests.Session.get", return_value=mock_get_response):
        dataframes_generator = controller.fetch_dataframes()
        dataframe = next(dataframes_generator, None)
        assert isinstance(dataframe, cudf.DataFrame)
        assert "link" in dataframe.columns
        assert len(dataframe) > 0


@pytest.mark.parametrize("strip_markup", [False, True])
@pytest.mark.parametrize("feed_input", [test_file_paths, test_file_paths[0]])
def test_fetch_dataframes_filepath(feed_input: str | list[str], strip_markup: bool):
    controller = RSSController(feed_input=feed_input, strip_markup=strip_markup)
    dataframes_generator = controller.fetch_dataframes()
    dataframe = next(dataframes_generator, None)
    assert isinstance(dataframe, cudf.DataFrame)
    assert "link" in dataframe.columns
    assert len(dataframe) > 0


@pytest.mark.parametrize("strip_markup", [False, True])
@pytest.mark.parametrize("feed_input, batch_size", [(test_file_paths, 5)])
def test_batch_size(feed_input: list[str], batch_size: int, strip_markup: bool):
    controller = RSSController(feed_input=feed_input, batch_size=batch_size, strip_markup=strip_markup)
    for df in controller.fetch_dataframes():
        assert isinstance(df, cudf.DataFrame)
        assert len(df) <= batch_size


@pytest.mark.parametrize("strip_markup", [False, True])
@pytest.mark.parametrize("feed_input, enable_cache", [(test_file_paths[0], False), (test_urls[0], True),
                                                      (test_urls[0], False)])
def test_try_parse_feed_with_beautiful_soup(feed_input: str,
                                            enable_cache: bool,
                                            mock_get_response: Mock,
                                            strip_markup: bool):
    controller = RSSController(feed_input=feed_input, enable_cache=enable_cache, strip_markup=strip_markup)

    # When enable_cache is set to 'True', the feed content is provided as input.
    feed_data = controller._try_parse_feed_with_beautiful_soup(mock_get_response.text)

    assert isinstance(feed_data, feedparser.FeedParserDict)
    assert len(feed_data.entries) > 0

    for entry in feed_data.entries:
        assert "title" in entry
        assert "link" in entry
        assert "id" in entry

        # Add more assertions as needed to validate the content of each entry
        for key, value in entry.items():
            if key not in ["title", "link", "id"]:
                assert value is not None

    # Additional assertions to validate the overall structure of the feed data
    assert isinstance(feed_data, dict)
    assert "entries" in feed_data
    assert isinstance(feed_data["entries"], list)


def test_parse_feeds(mock_feed: feedparser.FeedParserDict):
    feed_input = test_urls[0]
    cooldown_interval = 620
    controller = RSSController(feed_input=feed_input, enable_cache=False, cooldown_interval=cooldown_interval)

    with patch("morpheus.controllers.rss_controller.feedparser.parse") as mock_feedparser_parse:

        mock_feedparser_parse.return_value = mock_feed

        with patch.object(controller, '_try_parse_feed') as mock_try_parse_feed:
            dataframes_generator = controller.parse_feeds()
            next(dataframes_generator, None)
            feed_stats: FeedStats = controller.get_feed_stats(feed_input)
            assert feed_stats.last_try_result == "Success"
            assert feed_stats.failure_count == 0
            assert feed_stats.success_count == 1

            # Raise exception to test failure scenario
            mock_try_parse_feed.side_effect = Exception("SampleException")
            dataframes_generator = controller.parse_feeds()
            next(dataframes_generator, None)

            feed_stats: FeedStats = controller.get_feed_stats(feed_input)
            assert feed_stats.last_try_result == "Failure"
            assert feed_stats.failure_count == 1
            assert feed_stats.success_count == 1

            # Skip trying until cooldown period is met.
            dataframes_generator = controller.parse_feeds()
            next(dataframes_generator, None)

            feed_stats: FeedStats = controller.get_feed_stats(feed_input)
            assert feed_stats.last_try_result == "Failure"
            assert feed_stats.failure_count == 1
            assert feed_stats.success_count == 1

            # Resume trying after cooldown period
            with patch("time.time", return_value=time.time() + cooldown_interval):

                dataframes_generator = controller.parse_feeds()
                next(dataframes_generator, None)

                feed_stats: FeedStats = controller.get_feed_stats(feed_input)
                assert feed_stats.last_try_result == "Failure"
                assert feed_stats.failure_count == 2
                assert feed_stats.success_count == 1

        with pytest.raises(ValueError):
            controller.get_feed_stats("http://testfeed.com")


@pytest.mark.parametrize("strip_markup", [False, True])
@pytest.mark.parametrize("feed_input", [test_urls[0]])
def test_redundant_fetch(feed_input: str,
                         mock_feed: feedparser.FeedParserDict,
                         mock_get_response: Mock,
                         strip_markup: bool):

    controller = RSSController(feed_input=feed_input, strip_markup=strip_markup)
    mock_feedparser_parse = patch("morpheus.controllers.rss_controller.feedparser.parse")
    with mock_feedparser_parse, patch("requests.Session.get", return_value=mock_get_response) as mocked_session_get:
        mock_feedparser_parse.return_value = mock_feed
        dataframes_generator = controller.fetch_dataframes()
        next(dataframes_generator, None)
        assert mocked_session_get.call_count == 1


@pytest.mark.parametrize("strip_markup", [False, True])
def test_strip_markup(cisa_rss_feed: list[str], strip_markup: bool):
    # Construct expected data
    tree = ElementTree.parse(cisa_rss_feed[0])

    # feedparser will map the description field to the summary field
    description_tags = tree.findall('./channel/item/description')
    expected_summary_col = [(tag.text or "").strip() for tag in description_tags]

    if strip_markup:
        expected_summary_col = [
            BeautifulSoup(summary, features="html.parser").get_text() for summary in expected_summary_col
        ]

    controller = RSSController(feed_input=cisa_rss_feed, strip_markup=strip_markup)
    dataframes = list(controller.fetch_dataframes())

    # The length number of dataframes and rows should be the same regardless if strip_markup is True or False
    assert len(dataframes) == 1
    dataframe = dataframes[0]
    assert isinstance(dataframe, cudf.DataFrame)
    assert len(dataframe) == 10

    series: SeriesType = dataframe["summary"]
    assert (series.to_pandas().values == expected_summary_col).all()
