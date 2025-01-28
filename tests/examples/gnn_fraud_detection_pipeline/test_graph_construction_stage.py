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

import types
from io import StringIO

import pytest
import torch

import cudf

from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta

# pylint: disable=no-name-in-module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA enabled torch installation")
@pytest.mark.gpu_mode
class TestGraphConstructionStage:

    def test_constructor(self, config: Config, training_file: str):
        from stages.graph_construction_stage import FraudGraphConstructionStage
        stage = FraudGraphConstructionStage(config, training_file)
        assert isinstance(stage._training_data, cudf.DataFrame)

        # The training datafile contains many more columns than this, but these are the four columns
        # that are depended upon in the code
        assert {'client_node', 'index', 'fraud_label', 'merchant_node'}.issubset(stage._column_names)

    def test_process_message(self, dgl: types.ModuleType, config: Config, test_data: dict):
        from stages import graph_construction_stage
        df = test_data['df']
        expected_nodes = test_data['expected_nodes']
        expected_edges = test_data['expected_edges']

        # The stage wants a csv file from the first 5 rows
        training_data = StringIO(df.head(5).to_csv(index=False))
        stage = graph_construction_stage.FraudGraphConstructionStage(config, training_data)

        # Since we used the first 5 rows as the training data, send the second 5 as inference data
        meta = MessageMeta(cudf.DataFrame(df).tail(5))
        control_msg = ControlMessage()
        control_msg.payload(meta)

        fgmm = stage._process_message(control_msg)

        assert isinstance(fgmm, ControlMessage)
        assert fgmm.payload().count == 5

        assert isinstance(fgmm.get_metadata("graph"), dgl.DGLGraph)

        # Since the graph has a reverse edge for each edge, one edge comparison is enough.
        buy_edges = fgmm.get_metadata("graph").edges(etype='buy')
        sell_edges = fgmm.get_metadata("graph").edges(etype='sell')

        # expected edges, convert [(u,v)] format to [u, v] of DGL edge format.
        exp_buy_edges = [torch.LongTensor(e).cuda() for e in zip(*expected_edges['buy'])]
        exp_sell_edges = [torch.LongTensor(e).cuda() for e in zip(*expected_edges['sell'])]

        # Compare all edges types agree.
        assert all(exp_buy_edges[0] == buy_edges[0]) & all(exp_buy_edges[1] == buy_edges[1])
        assert all(exp_sell_edges[0] == sell_edges[0]) & all(exp_sell_edges[1] == sell_edges[1])

        # Compare nodes.
        for node in ['client', 'merchant']:
            assert fgmm.get_metadata("graph").nodes(node).tolist() == list(expected_nodes[node + "_node"])
