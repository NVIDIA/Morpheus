#!/usr/bin/env python
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

import os
from unittest import mock

import fsspec
import pytest

import morpheus.utils.downloader
from _utils import TEST_DIRS
from morpheus.utils.downloader import DOWNLOAD_METHODS_MAP
from morpheus.utils.downloader import Downloader
from morpheus.utils.downloader import DownloadMethods


@pytest.fixture(name="dask_distributed", autouse=True, scope='session')
def dask_distributed_fixture(dask_distributed):
    """
    Mark tests requiring dask.distributed
    """
    yield dask_distributed


@pytest.mark.usefixtures("restore_environ")
@pytest.mark.parametrize('use_env', [True, False])
@pytest.mark.parametrize('dl_method', ["single_thread", "dask", "dask_thread"])
def test_constructor_download_type(use_env: bool, dl_method: str):
    kwargs = {}
    if use_env:
        os.environ['MORPHEUS_FILE_DOWNLOAD_TYPE'] = dl_method
    else:
        kwargs['download_method'] = dl_method

    downloader = Downloader(**kwargs)
    assert downloader.download_method == DOWNLOAD_METHODS_MAP[dl_method]


@pytest.mark.parametrize('dl_method', list(DownloadMethods))
def test_constructor_enum_vals(dl_method: DownloadMethods):
    downloader = Downloader(download_method=dl_method)
    assert downloader.download_method == dl_method


@pytest.mark.usefixtures("restore_environ")
@pytest.mark.parametrize('dl_method', [DownloadMethods.DASK, DownloadMethods.DASK_THREAD])
def test_constructor_env_wins(dl_method: DownloadMethods):
    os.environ['MORPHEUS_FILE_DOWNLOAD_TYPE'] = "single_thread"
    downloader = Downloader(download_method=dl_method)
    assert downloader.download_method == DownloadMethods.SINGLE_THREAD


@pytest.mark.usefixtures("restore_environ")
@pytest.mark.parametrize('use_env', [True, False])
def test_constructor_invalid_dltype(use_env: bool):
    kwargs = {}
    if use_env:
        os.environ['MORPHEUS_FILE_DOWNLOAD_TYPE'] = "fake"
    else:
        kwargs['download_method'] = "fake"

    with pytest.raises(ValueError):
        Downloader(**kwargs)


@pytest.mark.reload_modules(morpheus.utils.downloader)
@pytest.mark.parametrize("dl_method", ["dask", "dask_thread"])
@pytest.mark.usefixtures("reload_modules")
@mock.patch('dask.distributed.LocalCluster')
def test_get_dask_cluster(mock_dask_cluster: mock.MagicMock, dl_method: str):
    mock_dask_cluster.return_value = mock_dask_cluster
    downloader1 = Downloader(download_method=dl_method)
    assert downloader1.get_dask_cluster() is mock_dask_cluster

    # create another downloader then assert that cluster was only created once
    downloader2 = Downloader(download_method=dl_method)
    downloader2.get_dask_cluster()
    assert downloader2.get_dask_cluster() is mock_dask_cluster

    mock_dask_cluster.assert_called_once()


@pytest.mark.reload_modules(morpheus.utils.downloader)
@pytest.mark.parametrize('dl_method', ["dask", "dask_thread"])
@pytest.mark.usefixtures("reload_modules")
@mock.patch('dask.distributed.LocalCluster')
def test_close(mock_dask_cluster: mock.MagicMock, dl_method: str):
    mock_dask_cluster.return_value = mock_dask_cluster
    downloader = Downloader(download_method=dl_method)
    assert downloader.get_dask_cluster() is mock_dask_cluster

    mock_dask_cluster.close.assert_not_called()
    downloader.close()


@mock.patch('dask.distributed.LocalCluster')
@pytest.mark.parametrize('dl_method', ["single_thread"])
def test_close_noop(mock_dask_cluster: mock.MagicMock, dl_method: str):
    mock_dask_cluster.return_value = mock_dask_cluster
    downloader = Downloader(download_method=dl_method)

    # Method is a no-op when Dask is not used
    downloader.close()

    mock_dask_cluster.assert_not_called()
    mock_dask_cluster.close.assert_not_called()


@pytest.mark.reload_modules(morpheus.utils.downloader)
@pytest.mark.usefixtures("reload_modules", "restore_environ")
@pytest.mark.parametrize('dl_method', ["single_thread", "dask", "dask_thread"])
@mock.patch('dask.config')
@mock.patch('dask.distributed.Client')
@mock.patch('dask.distributed.LocalCluster')
def test_download(mock_dask_cluster: mock.MagicMock,
                  mock_dask_client: mock.MagicMock,
                  mock_dask_config: mock.MagicMock,
                  dl_method: str):
    mock_dask_config.get = lambda key: 1.0 if (key == "distributed.comm.timesouts.connect") else None
    mock_dask_cluster.return_value = mock_dask_cluster
    mock_dask_client.return_value = mock_dask_client
    mock_dask_client.__enter__.return_value = mock_dask_client
    mock_dask_client.__exit__.return_value = False

    input_glob = os.path.join(TEST_DIRS.tests_data_dir, 'appshield/snapshot-1/*.json')
    download_buckets = fsspec.open_files(input_glob)
    num_buckets = len(download_buckets)
    assert num_buckets > 0

    download_fn = mock.MagicMock()
    returnd_df = mock.MagicMock()
    if dl_method.startswith('dask'):
        mock_dask_client.gather.return_value = [returnd_df for _ in range(num_buckets)]
    else:
        download_fn.return_value = returnd_df

    downloader = Downloader(download_method=dl_method)

    results = downloader.download(download_buckets, download_fn)

    if dl_method == "single_thread":
        download_fn.assert_has_calls([mock.call(bucket) for bucket in download_buckets])
    else:
        download_fn.assert_not_called()

    if dl_method.startswith('dask'):
        mock_dask_client.assert_called_once_with(mock_dask_cluster)
        mock_dask_client.map.assert_called_once()
        mock_dask_client.gather.assert_called_once()
    else:
        mock_dask_cluster.assert_not_called()
        mock_dask_client.assert_not_called()
        mock_dask_config.assert_not_called()

    assert results == [returnd_df for _ in range(num_buckets)]


@pytest.mark.usefixtures("restore_environ")
@pytest.mark.parametrize('use_env', [True, False])
@pytest.mark.parametrize('dl_method', ["multiprocess", "multiprocessing"])
def test_constructor_multiproc_dltype_not_supported(use_env: bool, dl_method: str):
    kwargs = {}
    if use_env:
        os.environ['MORPHEUS_FILE_DOWNLOAD_TYPE'] = dl_method
    else:
        kwargs['download_method'] = dl_method

    with pytest.raises(ValueError) as excinfo:
        Downloader(**kwargs)

    assert "no longer supported" in str(excinfo.value)
