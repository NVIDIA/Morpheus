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
"""
Downloader utility class for fetching files, potentially from a remote file service, using a variety of methods defined
by the `DownloadMethods` enum.
"""

import logging
import os
import threading
import typing
from enum import Enum

import fsspec
import pandas as pd

logger = logging.getLogger(__name__)


class DownloadMethods(str, Enum):
    """Valid download methods for the `Downloader` class."""
    SINGLE_THREAD = "single_thread"
    DASK = "dask"
    DASK_THREAD = "dask_thread"


DOWNLOAD_METHODS_MAP = {dl.value: dl for dl in DownloadMethods}


class Downloader:
    """
    Downloads a list of `fsspec.core.OpenFiles` files using one of the following methods:
        single_thread, dask or dask_thread

    The download method can be passed in via the `download_method` parameter or via the `MORPHEUS_FILE_DOWNLOAD_TYPE`
    environment variable. If both are set, the environment variable takes precedence, by default `dask_thread` is used.

    When using single_thread, `dask` and `dask.distributed` is not reuiqrred to be installed.

    Parameters
    ----------
    download_method : typing.Union[DownloadMethods, str], optional, default = DownloadMethods.DASK_THREAD
        The download method to use, if the `MORPHEUS_FILE_DOWNLOAD_TYPE` environment variable is set, it takes
        presedence.
    dask_heartbeat_interval : str, optional, default = "30s"
        The heartbeat interval to use when using dask or dask_thread.
    """

    # This cluster is shared by all Downloader instances that use dask download method.
    _dask_cluster = None

    _mutex = threading.RLock()

    def __init__(self,
                 download_method: typing.Union[DownloadMethods, str] = DownloadMethods.DASK_THREAD,
                 dask_heartbeat_interval: str = "30s"):

        self._dask_heartbeat_interval = dask_heartbeat_interval

        download_method = os.environ.get("MORPHEUS_FILE_DOWNLOAD_TYPE", download_method)

        if isinstance(download_method, str):
            if (download_method in ("multiprocess", "multiprocessing")):
                raise ValueError(
                    f"The '{download_method}' download method is no longer supported. Please use 'dask' or "
                    "'single_thread' instead.")
            try:
                download_method = DOWNLOAD_METHODS_MAP[download_method.lower()]
            except KeyError as exc:
                raise ValueError(
                    f"Invalid download method: {download_method}. Valid values are: {DOWNLOAD_METHODS_MAP.keys()}"
                ) from exc

        self._download_method = download_method

    @property
    def download_method(self) -> str:
        """Return the download method."""
        return self._download_method

    def get_dask_cluster(self):
        """
        Get the dask cluster used by this downloader. If the cluster does not exist, it is created.

        Returns
        -------
        dask.distributed.LocalCluster
        """

        with Downloader._mutex:
            if Downloader._dask_cluster is None:
                import dask
                import dask.distributed

                logger.debug("Creating dask cluster...")

                Downloader._dask_cluster = dask.distributed.LocalCluster(start=True,
                                                                         processes=self.download_method
                                                                         != "dask_thread")

                logger.debug("Creating dask cluster... Done. Dashboard: %s", Downloader._dask_cluster.dashboard_link)

            return Downloader._dask_cluster

    def get_dask_client(self):
        """
        Construct a dask client using the cluster created by `get_dask_cluster`

        Returns
        -------
        dask.distributed.Client
        """
        import dask.distributed
        return dask.distributed.Client(self.get_dask_cluster())

    def close(self):
        """Close the dask cluster if it exists."""
        if (self._dask_cluster is not None):
            logger.debug("Stopping dask cluster...")

            self._dask_cluster.close()

            self._dask_cluster = None

            logger.debug("Stopping dask cluster... Done.")

    def download(self,
                 download_buckets: fsspec.core.OpenFiles,
                 download_fn: typing.Callable[[fsspec.core.OpenFile], pd.DataFrame]) -> typing.List[pd.DataFrame]:
        """
        Download the files in `download_buckets` using the method specified in the constructor.
        If dask or dask_thread is used, the `get_dask_client_fn` function is used to create a dask client, otherwise
        it is not called. If using one of the other methods this can be set to None.

        Parameters
        ----------
        download_buckets : typing.Iterable[fsspec.core.OpenFiles]
            Files to download
        download_fn : typing.Callable[[fsspec.core.OpenFiles], pd.DataFrame]
            Function used to download an individual file and return the contents as a pandas DataFrame

        Returns
        -------
        typing.List[pd.DataFrame]
        """
        dfs = []
        if (self._download_method.startswith("dask")):
            # Create the client each time to ensure all connections to the cluster are closed (they can time out)
            with self.get_dask_client() as dist:
                dfs = dist.map(download_fn, download_buckets)
                dfs = dist.gather(dfs)

        else:
            # Simply loop
            for open_file in download_buckets:
                dfs.append(download_fn(open_file))

        return dfs
