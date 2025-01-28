# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
import time

import pandas as pd

logger = logging.getLogger(__name__)

IMPORT_EXCEPTION = None
IMPORT_ERROR_MESSAGE = "ElasticsearchController requires the elasticsearch package to be installed."

try:
    from elasticsearch import ConnectionError as ESConnectionError
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import parallel_bulk
except ImportError as import_exc:
    IMPORT_EXCEPTION = import_exc


class ElasticsearchController:
    """
    ElasticsearchController to perform read and write operations using Elasticsearch service.

    Parameters
    ----------
    connection_kwargs : dict
        Keyword arguments to configure the Elasticsearch connection.
    raise_on_exception : bool, optional, default: False
        Whether to raise exceptions on Elasticsearch errors.
    refresh_period_secs : int, optional, default: 2400
        The refresh period in seconds for client refreshing.
    """

    def __init__(self, connection_kwargs: dict, raise_on_exception: bool = False, refresh_period_secs: int = 2400):
        if IMPORT_EXCEPTION is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE) from IMPORT_EXCEPTION

        self._client = None
        self._last_refresh_time = None
        self._raise_on_exception = raise_on_exception
        self._refresh_period_secs = refresh_period_secs

        if connection_kwargs is not None and not connection_kwargs:
            raise ValueError("Connection kwargs cannot be none or empty.")

        self._connection_kwargs = connection_kwargs

        logger.debug("Creating Elasticsearch client with configuration: %s", connection_kwargs)

        self.refresh_client(force=True)

        logger.debug("Elasticsearch cluster info: %s", self._client.info)
        logger.debug("Creating Elasticsearch client... Done!")

    def refresh_client(self, force: bool = False) -> bool:
        """
        Refresh the Elasticsearch client instance.

        Parameters
        ----------
        force : bool, optional, default: False
            Force a client refresh.

        Returns
        -------
        bool
            Returns true if client is refreshed, otherwise false.
        """

        is_refreshed = False
        time_now = time.time()
        if force or self._client is None or time_now - self._last_refresh_time >= self._refresh_period_secs:
            if self._client:
                try:
                    # Close the existing client
                    self.close_client()
                except Exception as ex:
                    logger.warning("Ignoring client close error: %s", ex)
            logger.debug("Refreshing Elasticsearch client....")

            # Create Elasticsearch client
            self._client = Elasticsearch(**self._connection_kwargs)

            # Check if the client is connected
            if self._client.ping():
                logger.debug("Elasticsearch client is connected.")
            else:
                raise ESConnectionError("Elasticsearch client is not connected.")

            logger.debug("Refreshing Elasticsearch client.... Done!")
            self._last_refresh_time = time.time()
            is_refreshed = True

        return is_refreshed

    def parallel_bulk_write(self, actions) -> None:
        """
        Perform parallel bulk writes to Elasticsearch.

        Parameters
        ----------
        actions : list
            List of actions to perform in parallel.
        """

        self.refresh_client()

        for success, info in parallel_bulk(self._client, actions=actions, raise_on_exception=self._raise_on_exception):
            if not success:
                logger.error("Error writing to ElasticSearch: %s", str(info))

    def search_documents(self, index: str, query: dict, **kwargs) -> dict:
        """
        Search for documents in Elasticsearch based on the given query.

        Parameters
        ----------
        index : str
            The name of the index to search.
        query : dict
            The DSL query for the search.
        **kwargs
            Additional keyword arguments that are supported by the Elasticsearch search method.

        Returns
        -------
        dict
            The search result returned by Elasticsearch.
        """

        try:
            self.refresh_client()
            result = self._client.search(index=index, query=query, **kwargs)
            return result
        except Exception as exc:
            logger.error("Error searching documents: %s", exc)
            if self._raise_on_exception:
                raise RuntimeError(f"Error searching documents: {exc}") from exc

            return {}

    def df_to_parallel_bulk_write(self, index: str, df: pd.DataFrame) -> None:
        """
        Converts DataFrames to actions and parallel bulk writes to Elasticsearch.

        Parameters
        ----------
        index : str
            The name of the index to write.
        df : pd.DataFrame
            DataFrame entries that require writing to Elasticsearch.
        """

        actions = [{"_index": index, "_source": row} for row in df.to_dict("records")]

        self.parallel_bulk_write(actions)  # Parallel bulk upload to Elasticsearch

    def close_client(self) -> None:
        """
        Close the Elasticsearch client connection.
        """
        self._client.close()
