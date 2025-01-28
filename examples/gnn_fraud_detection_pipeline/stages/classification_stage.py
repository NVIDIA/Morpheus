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

import cuml

from morpheus.cli.register_stage import register_stage
from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema


@register_stage("gnn-fraud-classification", modes=[PipelineModes.OTHER])
class ClassificationStage(SinglePortStage):

    def __init__(self, c: Config, model_xgb_file: str):
        """
        The GNN Fraud Classification Stage

        Parameters
        ----------
        c : Config
            The Morpheus global config object
        model_xgb_file : str
            The XGB model to load
        """

        super().__init__(c)

        self._xgb_model = cuml.ForestInference.load(model_xgb_file, output_class=True, model_type="xgboost")
        self._needed_columns.update({'node_id': TypeId.INT64, 'prediction': TypeId.FLOAT32})

    @property
    def name(self) -> str:
        return "gnn-fraud-classification"

    def accepted_types(self) -> typing.Tuple:
        return (ControlMessage, )

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(ControlMessage)

    def supports_cpp_node(self) -> bool:
        return False

    def _process_message(self, message: ControlMessage) -> ControlMessage:

        inductive_embedding_column_names = message.get_metadata("inductive_embedding_column_names")
        ind_emb_columns = message.payload().get_data(inductive_embedding_column_names)

        node_identifiers = message.get_metadata("node_identifiers")
        message.payload().set_data("node_id", node_identifiers)

        # The XGBoost model is returning two probabilities for the binary classification. The first (column 0) is
        # probability that the transaction is in the benign class, and the second (column 1) is the probability that
        # the transaction is in the fraudulent class. Added together the two values will always equal 1.
        prediction = self._xgb_model.predict_proba(ind_emb_columns).iloc[:, 1]

        message.payload().set_data("prediction", prediction)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self._process_message))
        builder.make_edge(input_node, node)
        return node
