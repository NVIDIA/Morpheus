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
"""
Usage example:
python inference.py --training-data ../../datasets/training-data/fraud-detection-training-data.csv\
        --validation-data ../../datasets/validation-data/fraud-detection-validation-data.csv\
        --model-dir ../../fraud-detection-models\
        --output-file out.txt --model-type HinSAGE

Note: This script uses symbolink of model file located at
../../../examples/gnn_fraud_detection_pipeline/stages/model.py
"""

import os

import click
import numpy as np
import pandas as pd
import torch
from model import HeteroRGCN
from model import HinSAGE
from model import build_fsi_graph
from model import load_model
from model import prepare_data

import cudf as cf
from cuml import ForestInference

np.random.seed(1001)
torch.manual_seed(1001)


@click.command()
@click.option('--training-data', help="Path to training data for graph structure.", default="data/training.csv")
@click.option('--validation-data', help="Path to validation data", default="data/validation.csv")
@click.option('--model-dir', help="path to model directory", default="modeldir")
@click.option('--target-node', help="Target node", default="transaction")
@click.option('--output-file', help="Path to csv inference result", default="out.csv")
@click.option('--model-type', help="Model type either RGCN/Graphsage", default="RGCN")
def main(training_data, validation_data, model_dir, target_node, output_file, model_type):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    meta_cols = ["client_node", "merchant_node", "index"]
    if model_type == "RGCN":
        gnn_model = HeteroRGCN
    else:
        gnn_model = HinSAGE

    # prepare data
    training_data, validation_data = cf.read_csv(training_data), cf.read_csv(validation_data)
    _, _, _, test_index, _, all_data = prepare_data(training_data, validation_data)

    # build graph structure
    input_graph, feature_tensors = build_fsi_graph(all_data, meta_cols)
    test_index = torch.from_dlpack(test_index.values.toDlpack()).long()

    # Load graph model, return only the gnn model
    model, _, _ = load_model(model_dir, gnn_model=gnn_model)

    # Load XGBoost model
    xgb_model = ForestInference.load(os.path.join(model_dir, 'xgb.pt'), output_class=True)

    model = model.to(device)
    input_graph = input_graph.to(device)

    # Perform inference
    test_embedding, test_seeds = model.inference(input_graph, feature_tensors.float(), test_index, target_node)

    # collect result . XGBoost predict_proba(test_embedding)[:, 1]
    #  indicates probability score of negative class using XGBoost.
    pred_score = xgb_model.predict_proba(test_embedding)[:, 1]
    df_result = pd.DataFrame(test_seeds.cpu(), columns=['node_id'])
    df_result['score'] = pred_score.get()

    df_result.to_csv(output_file, index=False)
    print(df_result)


if __name__ == '__main__':
    main()
