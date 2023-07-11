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
"""Usage example:
python inference.py --training-data data/training.csv
    --validation-data data/validation.csv
    --model-dir model_dir --output-file out.csv
"""

import os
import pickle

import click
import dgl
import numpy as np
import pandas as pd
import torch
from model import HeteroRGCN

np.random.seed(1001)
torch.manual_seed(1001)


def build_fsi_graph(train_data, col_drop):
    """Build heterograph from edglist and node index.

    Args:
        train_data (pd.DataFrame): training data for node features.
        col_drop (list): features to drop from node features.

    Returns:
       Tuple[DGLGraph, torch.tensor]: dgl graph, normalized feature tensor
    """

    edge_list = {
        ('client', 'buy', 'transaction'): (train_data['client_node'].values, train_data['index'].values),
        ('transaction', 'bought', 'client'): (train_data['index'].values, train_data['client_node'].values),
        ('transaction', 'issued', 'merchant'): (train_data['index'].values, train_data['merchant_node'].values),
        ('merchant', 'sell', 'transaction'): (train_data['merchant_node'].values, train_data['index'].values)
    }

    graph = dgl.heterograph(edge_list)
    feature_tensors = torch.from_numpy(train_data.drop(col_drop, axis=1).values).float()
    feature_tensors = (feature_tensors - feature_tensors.mean(0)) / (0.0001 + feature_tensors.std(0))

    return graph, feature_tensors


def map_node_id(df, col_name):
    """ Convert column node list to integer index for dgl graph.

    Args:
        df (pd.DataFrame): dataframe
        col_name (list) : column list
    """
    node_index = {j: i for i, j in enumerate(df[col_name].unique())}
    df[col_name] = df[col_name].map(node_index)


def prepare_data(training_data, test_data):
    """Process data for training/inference operation

    Parameters
    ----------
    training_data : str
        path to training data
    test_data : str
        path to test/validation data
    Returns
    -------
    tuple
     tuple of (training_data, test_data, train_index, test_index, label, combined data)
    """

    df_train = pd.read_csv(training_data)
    train_idx_ = df_train.shape[0]
    df_test = pd.read_csv(test_data)
    df = pd.concat([df_train, df_test], axis=0)
    df['tran_id'] = df['index']

    meta_cols = ['tran_id', 'client_node', 'merchant_node']
    for col in meta_cols:
        map_node_id(df, col)

    train_idx = df['tran_id'][:train_idx_]
    test_idx = df['tran_id'][train_idx_:]

    df['index'] = df['tran_id']
    df.index = df['index']

    return (df.iloc[train_idx, :], df.iloc[test_idx, :], train_idx, test_idx, df['fraud_label'].values, df)


def load_model(model_dir, device):
    """Load trained model, graph structure from given directory

    Args:
        model_dir (str): directory path for trained model obj.
        device (str): device runtime.

    Returns:
        List[HeteroRGCN, DGLHeteroGraph]: model and graph structure.
    """
    from cuml import ForestInference

    with open(os.path.join(model_dir, "graph.pkl"), 'rb') as f:
        graph = pickle.load(f)
    with open(os.path.join(model_dir, 'hyperparams.pkl'), 'rb') as f:
        hyperparameters = pickle.load(f)
    model = HeteroRGCN(graph,
                       in_size=hyperparameters['in_size'],
                       hidden_size=hyperparameters['hidden_size'],
                       out_size=hyperparameters['out_size'],
                       n_layers=hyperparameters['n_layers'],
                       embedding_size=hyperparameters['embedding_size'],
                       target=hyperparameters['target_node'],
                       device=device)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt')))
    xgb_model = ForestInference.load(os.path.join(model_dir, 'xgb.pt'), output_class=True)

    return model, xgb_model, graph


@torch.no_grad()
def evaluate(model, eval_loader, feature_tensors, target_node, device='cpu'):
    """Takes trained RGCN model and input dataloader & produce logits and embedding.

    Args:
        model (HeteroRGCN): trained HeteroRGCN model object
        eval_loader (NodeDataLoader): evaluation dataloader
        feature_tensors (torch.Tensor) : test feature tensor
        target_node (str): target node encoding.
        device (str, optional): device runtime. Defaults to 'cpu'.

    Returns:
        List: logits, index & output embedding.
    """
    model.eval()
    eval_logits = []
    eval_seeds = []
    embedding = []

    for _, output_nodes, blocks in eval_loader:

        seed = output_nodes[target_node]

        nid = blocks[0].srcnodes[target_node].data[dgl.NID]
        blocks = [b.to(device) for b in blocks]
        input_features = feature_tensors[nid].to(device)
        logits, embedd = model.infer(blocks, input_features)
        eval_logits.append(logits.cpu().detach())
        eval_seeds.append(seed)
        embedding.append(embedd)

    eval_logits = torch.cat(eval_logits)
    eval_seeds = torch.cat(eval_seeds)
    embedding = torch.cat(embedding)
    return eval_logits, eval_seeds, embedding


def inference(model, input_graph, feature_tensors, test_idx, target_node, device):
    """Minibatch inference on test graph

    Args:
        model (HeteroRGCN) : trained HeteroRGCN model.
        input_graph (DGLHeterograph) : test graph
        feature_tensors (torch.Tensor) : node features
        test_idx (list): test index
        target_node (list): target node
        device (str, optional): device runtime.

    Returns:
        list: logits, index, output embedding
    """

    # create sampler and test dataloaders
    full_sampler = dgl.dataloading.MultiLayerNeighborSampler([4, 3])
    test_dataloader = dgl.dataloading.DataLoader(input_graph, {target_node: test_idx},
                                                 full_sampler,
                                                 batch_size=100,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 num_workers=0)
    test_logits, test_seeds, test_embedding = evaluate(
        model, test_dataloader, feature_tensors,  target_node, device=device)

    return test_logits, test_seeds, test_embedding


@click.command()
@click.option('--training-data', help="Path to training data for graph structure.", default="data/training.csv")
@click.option('--validation-data', help="Path to validation data", default="data/validation.csv")
@click.option('--model-dir', help="path to model directory", default="modeldir")
@click.option('--target-node', help="Target node", default="transaction")
@click.option('--output-file', help="Path to csv inference result", default="out.csv")
def main(training_data, validation_data, model_dir, target_node, output_file):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    meta_cols = ["client_node", "merchant_node", "fraud_label", "index", "tran_id"]

    # prepare data
    _, _, _, test_idx, _, all_data = prepare_data(training_data, validation_data)

    # build graph structure
    g_test, feature_tensors = build_fsi_graph(all_data, meta_cols)

    # Load graph model
    model, xgb_model, _ = load_model(model_dir, device)
    model = model.to(device)
    g_test = g_test.to(device)
    feature_tensors = feature_tensors.to(device)
    test_idx = torch.from_numpy(test_idx.values).to(device)

    _, test_seeds, test_embedding = inference(model, g_test, feature_tensors, test_idx, target_node, device)

    # collect result
    pred_score = xgb_model.predict_proba(test_embedding)[:, 1]
    df_result = pd.DataFrame(test_seeds.cpu(), columns=['node_id'])
    df_result['score'] = pred_score.get()

    df_result.to_csv(output_file, index=False)


if __name__ == '__main__':

    main()
