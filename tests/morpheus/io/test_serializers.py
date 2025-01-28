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

import typing
from io import BytesIO

import pandas as pd
import pytest

from _utils.dataset_manager import DatasetManager
from morpheus.common import FileTypes
from morpheus.io.serializers import df_to_csv
from morpheus.io.serializers import df_to_json
from morpheus.io.serializers import df_to_parquet
from morpheus.io.serializers import df_to_stream_csv
from morpheus.io.serializers import df_to_stream_json
from morpheus.io.serializers import df_to_stream_parquet


@pytest.mark.parametrize(
    "serialize_fn,file_type,is_stream,serialize_kwargs",
    [(df_to_csv, FileTypes.CSV, False, {
        'include_header': True, 'include_index_col': False
    }), (df_to_json, FileTypes.JSON, False, {}), (df_to_parquet, FileTypes.PARQUET, False, {}),
     (df_to_stream_csv, FileTypes.CSV, True, {
         'include_header': True, 'include_index_col': False
     }), (df_to_stream_json, FileTypes.JSON, True, {
         'lines': False
     }), (df_to_stream_json, FileTypes.JSON, True, {
         'lines': True
     }), (df_to_stream_parquet, FileTypes.PARQUET, True, {})],
    ids=['csv', 'jsonlines', 'parquet', 'stream_csv', 'stream_json', 'stream_jsonlines', 'stream_parquet'])
def test_serializers(dataset: DatasetManager,
                     serialize_fn: typing.Callable,
                     file_type: FileTypes,
                     is_stream: bool,
                     serialize_kwargs: dict):
    df = dataset['filter_probs.csv']

    serialize_kwargs['df'] = df

    if is_stream:
        serialize_kwargs['stream'] = BytesIO()

    results = serialize_fn(**serialize_kwargs)

    if is_stream:
        assert results is serialize_kwargs['stream']
        results.seek(0)
    else:
        if isinstance(results[0], str):
            results = [row.encode('utf-8') for row in results]

        results = BytesIO(initial_bytes=b"".join(results))

    deserialize_kwargs = {}
    if file_type == FileTypes.CSV:
        deserialize_fn = pd.read_csv
    elif file_type == FileTypes.JSON:
        deserialize_fn = pd.read_json
        deserialize_kwargs.update({'lines': serialize_kwargs.get('lines', True), 'orient': 'records'})
    elif file_type == FileTypes.PARQUET:
        deserialize_fn = pd.read_parquet

    result_df = deserialize_fn(results, **deserialize_kwargs)

    dataset.assert_compare_df(df, result_df)
