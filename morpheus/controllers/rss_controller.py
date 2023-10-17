# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
import os
import typing
from urllib.parse import urlparse

import feedparser
import requests_cache

import cudf

logger = logging.getLogger(__name__)


class RSSController:
    """
    RSSController handles fetching and processing of RSS feed entries.

    Parameters
    ----------
    feed_input : str, list[str]
        The URL or file path of the RSS feed.
    batch_size : int, optional, default = 128
        Number of feed items to accumulate before creating a DataFrame.
    run_indefinitely : bool, optional
        Whether to run the processing indefinitely. If set to True, the controller will continue fetching and processing
        If set to False, the controller will stop processing after the feed is fully fetched and processed.
        If not provided any value and if `feed_input` is of type URL, the controller will run indefinitely.
        Default is None.
    """

    def __init__(self, feed_input: str | list[str], batch_size: int = 128, run_indefinitely: bool = None):

        if (isinstance(feed_input, str)):
            feed_input = [feed_input]

        # Convert list to set to remove any duplicate feed inputs.
        self._feed_input = set(feed_input)
        self._batch_size = batch_size
        self._previous_entries = set()  # Stores the IDs of previous entries to prevent the processing of duplicates.

        if (run_indefinitely is None):
            # If feed_input is URL. Runs indefinitely
            run_indefinitely = any(RSSController.is_url(f) for f in self._feed_input)

        self._run_indefinitely = run_indefinitely

        self._session = requests_cache.CachedSession(os.path.join("./.cache/http", "RSSController.sqlite"),
                                                     backend="sqlite")

        self._errored_feeds = []  # Feeds that have thrown an error and wont be retried

    @property
    def run_indefinitely(self):
        """Property that determines to run the source indefinitely"""
        return self._run_indefinitely


    def _try_parse_feed(self, url: str):
        is_url = RSSController.is_url(url)

        fallback = False
        cache_hit = False

        if is_url:
            response = self._session.get(url)
            cache_hit = response.from_cache
            feed_input = response.text
        else:
            feed_input = url

        feed = feedparser.parse(feed_input)

        if feed["bozo"]:
            cache_hit = False

            if is_url:
                fallback = True
                try:
                    logger.info(f"Failed to parse feed: {url}. Trying to parse using feedparser directly.")
                    feed = feedparser.parse(url)
                except Exception as ex:
                    raise RuntimeError(f"Failed to parse feed using fallback: {url}: {ex}") from ex

            if feed["bozo"]:
                raise RuntimeError(f"Invalid feed input: {url}. Error: {feed['bozo_exception']}")

        logger.debug("Parsed feed: %s. Cache hit: %s. Fallback: %s", url, cache_hit, fallback)

        return feed


    def parse_feeds(self):
        """
        Parse the RSS feed using the feedparser library.

        Returns
        -------
        feedparser.FeedParserDict
            The parsed feed content.

        Raises
        ------
        RuntimeError
            If the feed input is invalid or does not contain any entries.
        """
        for url in self._feed_input:
            try:
                if (url in self._errored_feeds):
                    continue

                feed = self._try_parse_feed(url)

                if (not feed.entries):
                    raise RuntimeError(f"Invalid feed input: {url}. No entries found.")

                yield feed

            except Exception as ex:
                logger.warning("Failed to parse feed: %s: %s. The feed will be not be retried.", url, ex)

                self._errored_feeds.append(url)

    def fetch_dataframes(self):
        """
        Fetch and process RSS feed entries.

        Yeilds
        -------
        typing.Union[typing.List[typing.Tuple], typing.List]
            List of feed entries or None if no new entries are available.

        Raises
        ------
        RuntimeError
            If there is error fetching or processing feed entries.
        """
        entry_accumulator = []
        current_entries = set()

        try:

            for feed in self.parse_feeds():

                for entry in feed.entries:
                    entry_id = entry.get('id')
                    current_entries.add(entry_id)
                    if entry_id not in self._previous_entries:
                        entry_accumulator.append(entry)

                        if self._batch_size > 0 and len(entry_accumulator) >= self._batch_size:
                            yield self.create_dataframe(entry_accumulator)
                            entry_accumulator.clear()

            self._previous_entries = current_entries

            # Yield any remaining entries.
            if entry_accumulator:
                df = self.create_dataframe(entry_accumulator)
                yield df
            else:
                logger.debug("No new entries found.")

        except Exception as exc:
            raise RuntimeError(f"Error fetching or processing feed entries: {exc}") from exc

    def create_dataframe(self, entries: typing.List[typing.Tuple]) -> cudf.DataFrame:
        """
        Create a DataFrame from accumulated entry data.

        Parameters
        ----------
        entries : typing.List[typing.Tuple]
            List of accumulated feed entries.

        Returns
        -------
        cudf.DataFrame
            A DataFrame containing feed entry data.

        Raises
        ------
        RuntimeError
            Error creating DataFrame.
        """
        try:
            return cudf.DataFrame(entries)
        except Exception as exc:
            logger.error("Error creating DataFrame: %s", exc)
            raise RuntimeError(f"Error creating DataFrame: {exc}") from exc

    @classmethod
    def is_url(cls, feed_input: str) -> bool:
        """
        Check if the provided input is a valid URL.

        Parameters
        ----------
        feed_input : str
            The input string to be checked.

        Returns
        -------
        bool
            True if the input is a valid URL, False otherwise.
        """
        try:
            parsed_url = urlparse(feed_input)
            return parsed_url.scheme != '' and parsed_url.netloc != ''
        except Exception:
            return False
