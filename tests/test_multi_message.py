#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from morpheus.config import CppConfig
from morpheus.messages.message_meta import MessageMeta
from morpheus.messages.multi_message import MultiMessage


@pytest.mark.parametrize('df_type', ['cudf', 'pandas'])
def test_copy_ranges(config, filter_probs_df, df_type):
    if CppConfig.get_should_use_cpp() and df_type == 'pandas':
        return

    if df_type == 'pandas':
        df = filter_probs_df.to_pandas()
    else:
        df = filter_probs_df

    meta = MessageMeta(df)
    assert meta.count == len(df)

    mm = MultiMessage(meta, 0, len(df))
    assert mm.meta.count == len(df)
    assert len(mm.get_meta()) == len(df)

    mm2 = mm.copy_ranges([(2, 6)])
    assert len(mm2.meta.df) == 4
    assert mm2.meta.count == 4
    assert len(mm2.get_meta()) == 4
    assert mm2.meta is not meta
    assert mm2.meta.df is not df

    # slice two different ranges of rows
    mm3 = mm.copy_ranges([(2, 6), (12, 15)])
    assert len(mm3.meta.df) == 7
    assert mm3.meta.count == 7
    assert len(mm3.get_meta()) == 7
    assert mm3.meta is not meta
    assert mm3.meta is not mm2.meta
    assert mm3.meta.df is not df
    assert mm3.meta.df is not mm2.meta.df


def test_set_meta(config, filter_probs_df):
    meta = MessageMeta(filter_probs_df)
    mm = MultiMessage(meta, 0, meta.count)

    mm2 = mm.copy_ranges([(2, 6), (12, 15)])
    assert len(mm2.get_meta()) == 7

    values = list(range(7))
    mm2.set_meta('v2', values)

    assert mm2.get_meta_list('v2') == values

    assert mm2.get_meta_list(None) == mm2.get_meta().to_arrow().to_pylist()
