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

import glob
import json
import os

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from _utils import TEST_DIRS
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.stages.input.appshield_source_stage import AppShieldSourceStage
from morpheus.utils.directory_watcher import DirectoryWatcher


@pytest.mark.parametrize('cols_include',
                         [[
                             'Base',
                             'Block',
                             'CommitCharge',
                             'End VPN',
                             'File',
                             'GrantedAccess',
                             'HandleValue',
                             'InInit',
                             'InLoad',
                             'InMem',
                             'Name',
                             'Offset',
                             'PID',
                             'Parent',
                             'Path',
                             'PrivateMemory',
                             'Process',
                             'Protection',
                             'SHA256',
                             'Size',
                             'Start VPN',
                             'State',
                             'TID',
                             'Tag',
                             'Type',
                             'Value',
                             'Variable',
                             'WaitReason',
                             'plugin',
                             'snapshot_id',
                             'timestamp'
                         ]])
@pytest.mark.parametrize('plugins_include', [['ldrmodules', 'threadlist', 'envars', 'vadinfo', 'handles']])
def test_constructor(tmp_path, config, plugins_include, cols_include):
    input_glob = os.path.join(tmp_path, '*', '*.json')

    source = AppShieldSourceStage(config, input_glob, plugins_include, cols_include)

    assert source._plugins_include == ['ldrmodules', 'threadlist', 'envars', 'vadinfo', 'handles']
    assert source._cols_include == [
        'Base',
        'Block',
        'CommitCharge',
        'End VPN',
        'File',
        'GrantedAccess',
        'HandleValue',
        'InInit',
        'InLoad',
        'InMem',
        'Name',
        'Offset',
        'PID',
        'Parent',
        'Path',
        'PrivateMemory',
        'Process',
        'Protection',
        'SHA256',
        'Size',
        'Start VPN',
        'State',
        'TID',
        'Tag',
        'Type',
        'Value',
        'Variable',
        'WaitReason',
        'plugin',
        'snapshot_id',
        'timestamp'
    ]
    assert source._cols_exclude == ['SHA256']
    assert source.name == 'from-appshield'
    assert isinstance(source._watcher, DirectoryWatcher)


@pytest.mark.parametrize('cols_include', [['a', 'b', 'c', 'd']])
@pytest.mark.parametrize(
    'input_df',
    [pd.DataFrame({
        'a': range(10), 'b': range(10, 20), 'c': range(1, 11)
    })],
)
def test_fill_interested_cols(cols_include, input_df):
    output_df = AppShieldSourceStage.fill_interested_cols(input_df, cols_include)
    actual_columns = list(output_df.columns)
    expected_columns = cols_include

    assert actual_columns == expected_columns


@pytest.mark.parametrize('cols_exclude', [['Block', 'Variable', 'Value']])
@pytest.mark.parametrize(
    'expected_df',
    [
        pd.DataFrame({
            'PID': ['304', '304', '304', '444', '444'],
            'Process': ['smss.exe', 'smss.exe', 'smss.exe', 'csrss.exe', 'csrss.exe']
        })
    ],
)
def test_read_file_to_df(cols_exclude, expected_df):
    input_file = os.path.join(TEST_DIRS.tests_data_dir,
                              'appshield',
                              'snapshot-1',
                              'envars_2022-01-30_10-26-01.017250.json')
    with open(input_file, 'r', encoding='latin1') as file:
        output_df = AppShieldSourceStage.read_file_to_df(file, cols_exclude)

    assert list(output_df.columns) == ['PID', 'Process']
    assert_frame_equal(output_df, expected_df)


@pytest.mark.parametrize('cols_exclude', [['Block', 'Variable', 'Value']])
@pytest.mark.parametrize(
    'expected_df',
    [
        pd.DataFrame({
            'PID': ['304', '304', '304', '444', '444'],
            'Process': ['smss.exe', 'smss.exe', 'smss.exe', 'csrss.exe', 'csrss.exe']
        })
    ],
)
@pytest.mark.parametrize('encoding', ['latin1'])
def test_load_df(cols_exclude, expected_df, encoding):
    input_file = os.path.join(TEST_DIRS.tests_data_dir,
                              'appshield',
                              'snapshot-1',
                              'envars_2022-01-30_10-26-01.017250.json')
    output_df = AppShieldSourceStage.load_df(input_file, cols_exclude, encoding)

    assert list(output_df.columns) == ['PID', 'Process']
    assert_frame_equal(output_df, expected_df)


@pytest.mark.parametrize('plugin', ['envars'])
@pytest.mark.parametrize('expected_new_columns', ['snapshot_id', 'timestamp', 'source', 'plugin'])
def test_load_meta_cols(plugin, expected_new_columns):
    input_file = os.path.join(TEST_DIRS.tests_data_dir,
                              'appshield',
                              'snapshot-1',
                              'envars_2022-01-30_10-26-01.017250.json')
    filepath_split = input_file.split('/')

    with open(input_file, 'r', encoding='latin1') as file:
        data = json.load(file)
    input_df = pd.DataFrame(columns=data['titles'], data=data['data'])
    output_df = AppShieldSourceStage.load_meta_cols(filepath_split, plugin, input_df)

    assert expected_new_columns in list(output_df.columns)
    assert output_df.snapshot_id.iloc[0] == 1
    assert output_df.plugin.iloc[0] == 'envars'
    assert output_df.source.iloc[0] == 'appshield'
    assert output_df.timestamp.iloc[0] == '2022-01-30_10-26-01.017250'


@pytest.mark.parametrize('input_dfs',
                         [[
                             pd.DataFrame({
                                 'PID': ['304', '304', '304', '444', '444'],
                                 'Process': ['smss.exe', 'smss.exe', 'smss.exe', 'csrss.exe', 'csrss.exe'],
                                 'source': ['appshield', 'appshield', 'appshield-v2', 'appshield', 'appshield-v2']
                             }),
                             pd.DataFrame({
                                 'PID': ['350', '360', '304', '563', '673'],
                                 'Process': ['smss.exe', 'smss.exe', 'smss.exe', 'csrss.exe', 'csrss.exe'],
                                 'source': ['appshield', 'appshield', 'appshield-v2', 'appshield', 'appshield-v2']
                             })
                         ]])
@pytest.mark.parametrize(
    'expected_appshield_df',
    [
        pd.DataFrame({
            'PID':
                pd.Series(['304', '304', '444', '350', '360', '563'], index=[0, 1, 3, 0, 1, 3]),
            'Process':
                pd.Series(['smss.exe', 'smss.exe', 'csrss.exe', 'smss.exe', 'smss.exe', 'csrss.exe'],
                          index=[0, 1, 3, 0, 1, 3]),
            'source':
                pd.Series(['appshield', 'appshield', 'appshield', 'appshield', 'appshield', 'appshield'],
                          index=[0, 1, 3, 0, 1, 3])
        }),
    ])
@pytest.mark.parametrize('source_col', ['source'])
def test_batch_source_split(input_dfs, source_col, expected_appshield_df):
    output_df_per_source = AppShieldSourceStage.batch_source_split(input_dfs, source_col)

    assert len(output_df_per_source) == 2
    assert len(output_df_per_source['appshield']) == 6
    assert len(output_df_per_source['appshield-v2']) == 4
    assert_frame_equal(output_df_per_source['appshield'], expected_appshield_df)


@pytest.mark.parametrize('cols_include',
                         [[
                             'Base',
                             'Block',
                             'CommitCharge',
                             'End VPN',
                             'File',
                             'GrantedAccess',
                             'HandleValue',
                             'InInit',
                             'InLoad',
                             'InMem',
                             'Name',
                             'Offset',
                             'PID',
                             'Parent',
                             'Path',
                             'PrivateMemory',
                             'Process',
                             'Protection',
                             'SHA256',
                             'Size',
                             'Start VPN',
                             'State',
                             'TID',
                             'Tag',
                             'Type',
                             'Value',
                             'Variable',
                             'WaitReason',
                             'plugin',
                             'snapshot_id',
                             'timestamp'
                         ]])
@pytest.mark.parametrize('cols_exclude', [['SHA256']])
@pytest.mark.parametrize('plugins_include', [['ldrmodules', 'threadlist', 'envars', 'vadinfo', 'handles']])
@pytest.mark.parametrize('meta_columns', ['snapshot_id', 'timestamp', 'source', 'plugin'])
@pytest.mark.parametrize('encoding', ['latin1'])
def test_files_to_dfs(cols_include, cols_exclude, plugins_include, meta_columns, encoding):
    input_glob = os.path.join(TEST_DIRS.tests_data_dir, 'appshield', 'snapshot-1', '*.json')
    file_list = glob.glob(input_glob)
    output_df_per_source = AppShieldSourceStage.files_to_dfs(file_list,
                                                             cols_include,
                                                             cols_exclude,
                                                             plugins_include,
                                                             encoding)

    assert len(output_df_per_source) == 1
    assert 'appshield' in output_df_per_source
    assert meta_columns in output_df_per_source['appshield'].columns


@pytest.mark.parametrize(
    'input_df_per_source',
    [{
        'appshield':
            pd.DataFrame({
                'PID':
                    pd.Series(['304', '304', '444', '350', '360', '563'], index=[0, 1, 3, 0, 1, 3]),
                'Process':
                    pd.Series(['smss.exe', 'smss.exe', 'csrss.exe', 'smss.exe', 'smss.exe', 'csrss.exe'],
                              index=[0, 1, 3, 0, 1, 3]),
                'source':
                    pd.Series(['appshield', 'appshield', 'appshield', 'appshield', 'appshield', 'appshield'],
                              index=[0, 1, 3, 0, 1, 3])
            }),
        'appshield-v2':
            pd.DataFrame({
                'PID':
                    pd.Series(['304', '304', '444', '350', '360', '563'], index=[0, 1, 3, 0, 1, 3]),
                'Process':
                    pd.Series(['smss.exe', 'smss.exe', 'csrss.exe', 'smss.exe', 'smss.exe', 'csrss.exe'],
                              index=[0, 1, 3, 0, 1, 3]),
                'source':
                    pd.Series([
                        'appshield-v2', 'appshield-v2', 'appshield-v2', 'appshield-v2', 'appshield-v2', 'appshield-v2'
                    ],
                              index=[0, 1, 3, 0, 1, 3])
            })
    }])
def test_build_messages(config: Config, tmp_path: str, input_df_per_source: dict):
    expected_sources = sorted(input_df_per_source.keys())

    input_glob = os.path.join(tmp_path, '*.json')
    # These constructor arguments are not used by the _build_messages method
    stage = AppShieldSourceStage(config, input_glob, ['unused'], ['unused'])
    appshield_messages = stage._build_messages(input_df_per_source)

    assert len(appshield_messages) == len(expected_sources)

    actual_sources = []
    for message in appshield_messages:
        assert isinstance(message, ControlMessage)
        actual_sources.append(message.get_metadata('source'))

    assert sorted(actual_sources) == expected_sources
