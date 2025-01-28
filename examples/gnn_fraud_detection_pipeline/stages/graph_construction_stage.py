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

import pathlib
import typing

import mrc
import torch
from mrc.core import operators as ops

import cudf

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema

from .model import build_fsi_graph
from .model import prepare_data


@register_stage("fraud-graph-construction", modes=[PipelineModes.OTHER])
class FraudGraphConstructionStage(SinglePortStage):

    def __init__(self, config: Config, training_file: pathlib.Path):
        """
        Create a fraud-graph-construction stage

        Parameters
        ----------
        c : Config
            The Morpheus config object
        training_file : pathlib.Path, exists = True, dir_okay = False
            A CSV training file to load to seed the graph
        """
        super().__init__(config)
        self._training_data = cudf.read_csv(training_file)
        self._column_names = self._training_data.columns.values.tolist()

    @property
    def name(self) -> str:
        return "fraud-graph-construction"

    def accepted_types(self) -> typing.Tuple:
        return (ControlMessage, )

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(ControlMessage)

    def supports_cpp_node(self) -> bool:
        return False

    def _process_message(self, message: ControlMessage) -> ControlMessage:

        _, _, _, test_index, _, graph_data = prepare_data(self._training_data,
                                                          message.payload().get_data(self._column_names))

        # meta columns to remove as node features
        meta_cols = ['client_node', 'merchant_node', 'index']
        graph, node_features = build_fsi_graph(graph_data, meta_cols)

        # Convert to torch.tensor from cupy
        test_index = torch.from_dlpack(test_index.values.toDlpack()).long()
        node_features = node_features.float()

        message.set_metadata("graph", graph)
        message.set_metadata("node_features", node_features.float())
        message.set_metadata("test_index", test_index)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self._process_message))
        builder.make_edge(input_node, node)
        return node
