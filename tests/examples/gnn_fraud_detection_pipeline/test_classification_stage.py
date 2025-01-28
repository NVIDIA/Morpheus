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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, eithe r express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import types

import pytest

from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta

# pylint: disable=no-name-in-module


@pytest.mark.gpu_mode
class TestClassificationStage:

    def test_constructor(self, config: Config, xgb_model: str, cuml: types.ModuleType):
        from stages.classification_stage import ClassificationStage

        stage = ClassificationStage(config, xgb_model)
        assert isinstance(stage._xgb_model, cuml.ForestInference)

    def test_process_message(self, config: Config, xgb_model: str, dataset_cudf: DatasetManager):
        from stages.classification_stage import ClassificationStage

        df = dataset_cudf['examples/gnn_fraud_detection_pipeline/inductive_emb.csv']
        df.rename(lambda x: f"ind_emb_{x}", axis=1, inplace=True)

        expected_df = dataset_cudf.pandas['examples/gnn_fraud_detection_pipeline/predictions.csv']
        assert len(df) == len(expected_df)

        # The exact values of the node_identifiers aren't important to this stage, we just need to verify that they're
        # inserted into a "node_id" column in the DF
        node_identifiers = expected_df['node_id'].tolist()

        ind_emb_columns = list(df.columns)

        meta = MessageMeta(df)
        msg = ControlMessage()
        msg.payload(meta)
        msg.set_metadata("node_identifiers", node_identifiers)
        msg.set_metadata("inductive_embedding_column_names", ind_emb_columns)

        stage = ClassificationStage(config, xgb_model)
        results = stage._process_message(msg)

        # The stage actually edits the message in place, and returns it, but we don't need to assert that
        dataset_cudf.assert_compare_df(results.payload().get_data(['prediction', 'node_id']), expected_df)
