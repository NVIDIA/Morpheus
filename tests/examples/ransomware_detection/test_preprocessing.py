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

import cupy as cp
import pandas as pd
import pytest

from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.stages.preprocess.preprocess_base_stage import PreprocessBaseStage


class TestPreprocessingRWStage:
    # pylint: disable=no-name-in-module

    def test_constructor(self, config: Config, rwd_conf: dict):
        from stages.preprocessing import PreprocessingRWStage

        stage = PreprocessingRWStage(config, feature_columns=rwd_conf['model_features'], sliding_window=6)
        assert isinstance(stage, PreprocessBaseStage)
        assert stage._feature_columns == rwd_conf['model_features']
        assert stage._features_len == len(rwd_conf['model_features'])
        assert not stage._snapshot_dict
        assert len(stage._padding_data) == len(rwd_conf['model_features']) * 6
        for i in stage._padding_data:
            assert i == 0

    def test_sliding_window_offsets(self, config: Config, rwd_conf: dict):
        from stages.preprocessing import PreprocessingRWStage

        stage = PreprocessingRWStage(config, feature_columns=rwd_conf['model_features'], sliding_window=6)

        window = 3
        ids = [17, 18, 19, 20, 21, 22, 23, 31, 32, 33]
        results = stage._sliding_window_offsets(ids, len(ids), window=window)
        assert results == [(0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (7, 10)]

    def test_sliding_window_non_consequtive(self, config: Config, rwd_conf: dict):
        # Non-consecutive ids don't create sliding windows
        from stages.preprocessing import PreprocessingRWStage

        stage = PreprocessingRWStage(config, feature_columns=rwd_conf['model_features'], sliding_window=6)

        window = 3
        ids = [17, 19, 21, 23, 31, 33]
        assert len(stage._sliding_window_offsets(list(reversed(ids)), len(ids), window=window)) == 0

    def test_sliding_window_offsets_errors(self, config: Config, rwd_conf: dict):
        from stages.preprocessing import PreprocessingRWStage

        stage = PreprocessingRWStage(config, feature_columns=rwd_conf['model_features'], sliding_window=6)

        # ids_len doesn't match the length of the ids list
        with pytest.raises(AssertionError):
            stage._sliding_window_offsets(ids=[5, 6, 7], ids_len=12, window=2)

        # Window is larger than ids
        with pytest.raises(AssertionError):
            stage._sliding_window_offsets(ids=[5, 6, 7], ids_len=3, window=4)

    def test_rollover_pending_snapshots(self, config: Config, rwd_conf: dict, dataset_pandas: DatasetManager):
        from stages.preprocessing import PreprocessingRWStage

        snapshot_ids = [5, 8, 10, 13]
        source_pid_process = "123_test.exe"
        df = dataset_pandas['examples/ransomware_detection/dask_results.csv']
        assert len(df) == len(snapshot_ids)

        # The snapshot_id's in the test data set are all '1', set them to different values
        df['snapshot_id'] = snapshot_ids
        df.index = df.snapshot_id

        stage = PreprocessingRWStage(config, feature_columns=rwd_conf['model_features'], sliding_window=4)
        stage._rollover_pending_snapshots(snapshot_ids, source_pid_process, df)

        assert list(stage._snapshot_dict.keys()) == [source_pid_process]

        # Due to the sliding window we should have all but the first snapshot_id in the results
        expected_snapshot_ids = snapshot_ids[1:]
        snapshots = stage._snapshot_dict[source_pid_process]

        assert len(snapshots) == len(expected_snapshot_ids)
        for (i, snapshot) in enumerate(snapshots):
            expected_snapshot_id = expected_snapshot_ids[i]
            assert snapshot.snapshot_id == expected_snapshot_id
            expected_data = df.loc[expected_snapshot_id].fillna('').values
            assert (pd.Series(snapshot.data).fillna('').values == expected_data).all()

    def test_rollover_pending_snapshots_empty_results(self,
                                                      config: Config,
                                                      rwd_conf: dict,
                                                      dataset_pandas: DatasetManager):
        from stages.preprocessing import PreprocessingRWStage

        snapshot_ids = []
        source_pid_process = "123_test.exe"
        df = dataset_pandas['examples/ransomware_detection/dask_results.csv']

        stage = PreprocessingRWStage(config, feature_columns=rwd_conf['model_features'], sliding_window=4)
        stage._rollover_pending_snapshots(snapshot_ids, source_pid_process, df)
        assert len(stage._snapshot_dict) == 0

    def test_merge_curr_and_prev_snapshots(self, config: Config, rwd_conf: dict, dataset_pandas: DatasetManager):
        from common.data_models import SnapshotData
        from stages.preprocessing import PreprocessingRWStage

        snapshot_ids = [5, 8, 10, 13]
        source_pid_process = "123_test.exe"
        df = dataset_pandas['examples/ransomware_detection/dask_results.csv']
        assert len(df) == len(snapshot_ids)
        df['snapshot_id'] = snapshot_ids
        df.index = df.snapshot_id

        stage = PreprocessingRWStage(config, feature_columns=rwd_conf['model_features'], sliding_window=4)
        test_row_8 = df.loc[8].copy(deep=True)
        test_row_8.pid_process = 'test_val1'

        test_row_13 = df.loc[13].copy(deep=True)
        test_row_13.pid_process = 'test_val2'

        stage._snapshot_dict = {
            source_pid_process: [SnapshotData(8, test_row_8.values), SnapshotData(13, test_row_13.values)]
        }

        expected_df = dataset_pandas['examples/ransomware_detection/dask_results.csv'].fillna('')
        expected_df.loc[1, 'pid_process'] = 'test_val1'
        expected_df.loc[3, 'pid_process'] = 'test_val2'

        expected_df.loc[:, 'snapshot_id'] = snapshot_ids
        expected_df.index = expected_df.snapshot_id

        stage._merge_curr_and_prev_snapshots(df, source_pid_process)
        dataset_pandas.assert_compare_df(df.fillna(''), expected_df)

    def test_pre_process_batch(self, config: Config, rwd_conf: dict, dataset_cudf: DatasetManager):
        from stages.preprocessing import PreprocessingRWStage
        df = dataset_cudf['examples/ransomware_detection/dask_results.csv']
        df['source_pid_process'] = 'appshield_' + df.pid_process
        expected_df = df.copy(deep=True).fillna('')
        meta = MessageMeta(df)
        cm = ControlMessage()
        cm.payload(meta)
        cm.set_metadata('source', 'tests')

        sliding_window = 4
        stage = PreprocessingRWStage(config, feature_columns=rwd_conf['model_features'], sliding_window=sliding_window)
        results: ControlMessage = stage._pre_process_batch(cm)
        assert isinstance(results, ControlMessage)

        expected_df['sequence'] = ['dummy' for _ in range(len(expected_df))]
        expected_input__0 = cp.asarray([0 for i in range(len(rwd_conf['model_features']) * sliding_window)])
        expected_seq_ids = cp.zeros((len(expected_df), 3), dtype=cp.uint32)
        expected_seq_ids[:, 0] = cp.arange(0, len(expected_df), dtype=cp.uint32)
        expected_seq_ids[:, 2] = len(rwd_conf['model_features']) * 3

        actual_df = results.payload().copy_dataframe().to_pandas().fillna('')
        dataset_cudf.assert_compare_df(actual_df, expected_df)

        actual_tensors = results.tensors()
        assert (actual_tensors.get_tensor('input__0') == expected_input__0).all()
        assert (actual_tensors.get_tensor('seq_ids') == expected_seq_ids).all()
