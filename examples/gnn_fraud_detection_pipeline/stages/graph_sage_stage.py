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

import mrc
from mrc.core import operators as ops

import cudf

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema

from .model import load_model


@register_stage("gnn-fraud-sage", modes=[PipelineModes.OTHER])
class GraphSAGEStage(SinglePortStage):

    def __init__(self,
                 config: Config,
                 model_dir: str,
                 batch_size: int = 100,
                 record_id: str = "index",
                 target_node: str = "transaction"):
        super().__init__(config)

        self._dgl_model, _, __ = load_model(model_dir)
        self._batch_size = batch_size
        self._record_id = record_id
        self._target_node = target_node

    @property
    def name(self) -> str:
        return "gnn-fraud-sage"

    def accepted_types(self) -> typing.Tuple:
        return (ControlMessage, )

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(ControlMessage)

    def supports_cpp_node(self) -> bool:
        return False

    def _process_message(self, message: ControlMessage) -> ControlMessage:

        node_identifiers = list(message.payload().get_data(self._record_id).to_pandas())

        # Perform inference
        inductive_embedding, _ = self._dgl_model.inference(message.get_metadata("graph"),
                                                           message.get_metadata("node_features"),
                                                           message.get_metadata("test_index"),
                                                           batch_size=self._batch_size)

        inductive_embedding = cudf.DataFrame(inductive_embedding)

        # Rename the columns to be more descriptive
        inductive_embedding.rename(lambda x: "ind_emb_" + str(x), axis=1, inplace=True)

        with message.payload().mutable_dataframe() as df:
            for col in inductive_embedding.columns.values.tolist():
                df[col] = inductive_embedding[col]

        assert (message.payload().count == len(inductive_embedding))

        message.set_metadata("node_identifiers", node_identifiers)
        message.set_metadata("inductive_embedding_column_names", inductive_embedding.columns.values.tolist())

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self._process_message))
        builder.make_edge(input_node, node)
        return node
