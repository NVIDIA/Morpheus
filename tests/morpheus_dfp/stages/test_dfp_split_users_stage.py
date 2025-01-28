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

import json
import os
import typing
import warnings
from collections import defaultdict

import pytest

from _utils import TEST_DIRS
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.type_utils import get_df_pkg_from_obj


def test_constructor(config: Config):
    from morpheus_dfp.stages.dfp_split_users_stage import DFPSplitUsersStage

    stage = DFPSplitUsersStage(config, include_generic=False, include_individual=True)
    assert isinstance(stage, SinglePortStage)
    assert not stage._include_generic
    assert stage._include_individual
    assert not stage._skip_users
    assert not stage._only_users
    assert not stage._user_index_map

    stage = DFPSplitUsersStage(config,
                               include_generic=True,
                               include_individual=False,
                               skip_users=['a', 'b'],
                               only_users=['c', 'd'])

    assert stage._include_generic
    assert not stage._include_individual
    assert stage._skip_users == ['a', 'b']
    assert stage._only_users == ['c', 'd']
    assert not stage._user_index_map


@pytest.mark.parametrize('include_generic', [True, False])
@pytest.mark.parametrize('include_individual', [True, False])
@pytest.mark.parametrize(
    'skip_users',
    [[], ['katherine-malton@forward.org'], ['terrietahon@planner-viral.com', 'angelaethridge@lomo-customer.net']])
@pytest.mark.parametrize('only_users',
                         [[], ['WENDY.HUERTA@Brooklyn-dynamic.edu'],
                          ['terrietahon@planner-viral.com', 'SAMUEL.DAVIS@transition-high-life.com']])
def test_extract_users(config: Config,
                       dataset: DatasetManager,
                       include_generic: bool,
                       include_individual: bool,
                       skip_users: typing.List[str],
                       only_users: typing.List[str]):
    from morpheus_dfp.stages.dfp_split_users_stage import DFPSplitUsersStage
    config.ae.userid_column_name = "From"
    config.ae.fallback_username = "testy_testerson"
    ts_col = config.ae.timestamp_column_name

    input_file = os.path.join(TEST_DIRS.tests_data_dir,
                              "examples/developer_guide/email_with_addresses_first_10.jsonlines")

    df = dataset[input_file]
    df_pkg = get_df_pkg_from_obj(df)

    # When the file is read using pandas (as is the case in the actual DFP pipeline), the timestamp column is
    # automatically converted to datetime objects. However cuDF doesn't do this and the column will contain integers.
    # When `dataset` is returning pandas DFs this might still be the case if `input_file` is first read using cuDF and
    # cached by the DatasetManager and then converted to pandas.
    if df[ts_col].dtype == 'int64':
        df[ts_col] = df_pkg.to_datetime(df[ts_col], unit='s')

    all_data = []
    expected_data = defaultdict(list)

    with open(input_file, encoding='UTF-8') as fh:
        for line in fh:
            json_data = json.loads(line)
            user_id = json_data['From']
            if user_id in skip_users:
                continue

            if len(only_users) > 0 and user_id not in only_users:
                continue

            json_data[ts_col] = df_pkg.to_datetime(json_data[ts_col], unit='s')

            if include_generic:
                all_data.append(json_data)

            if include_individual:
                expected_data[user_id].append(json_data)

    if include_generic:
        expected_data[config.ae.fallback_username] = all_data

    stage = DFPSplitUsersStage(config,
                               include_generic=include_generic,
                               include_individual=include_individual,
                               skip_users=skip_users,
                               only_users=only_users)

    with warnings.catch_warnings():
        # Ignore warning about the log message not being set. This happens whenever there aren't any output_messages
        warnings.filterwarnings("ignore",
                                message="Must set log msg before end of context! Skipping log",
                                category=UserWarning)
        results = stage.extract_users(df)

    if not include_generic and not include_individual:
        # Extra check for weird combination
        assert len(results) == 0

    # Add one for the generic user
    assert len(results) == len(expected_data)
    for msg in results:
        actual_df = msg.payload().df
        user_id = msg.get_metadata('user_id')
        assert len(actual_df) == len(expected_data[user_id])
        if user_id != config.ae.fallback_username:
            assert actual_df.to_dict('records') == expected_data[user_id]


def test_extract_users_none_to_empty(config: Config):
    from morpheus_dfp.stages.dfp_split_users_stage import DFPSplitUsersStage

    stage = DFPSplitUsersStage(config, include_generic=True, include_individual=True)
    assert not stage.extract_users(None)
