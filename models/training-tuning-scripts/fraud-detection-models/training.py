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
python training.py --training-data ../../datasets/training-data/fraud-detection-training-data.csv\
      --validation-data ../../datasets/validation-data/fraud-detection-validation-data.csv \
          --model-dir ../../fraud-detection-models --output-file out.txt\
            --model-type HinSAGE
"""
import os
import pickle

import click
import dgl
import numpy as np
import pandas as pd
import torch
from model import HeteroRGCN
from model import HinSAGE
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from torch import nn
from torchmetrics.functional import accuracy
from tqdm import trange
from xgboost import XGBClassifier

import cudf as cf
from cuml import ForestInference

np.random.seed(1001)
torch.manual_seed(1001)


def get_metrics(pred, labels, name='HinSAGE'):
    """Compute evaluation metrics.

    Parameters
    ----------
    pred : numpy.ndarray
        Predictions made by the model.
    labels : numpy.ndarray
        Groundtruth labels.
    name : str, optional
        Model name. Defaults to 'RGCN'.

    Returns
    -------
    List[List]
        List of evaluation metrics including:
        - f1: F1-score
        - precision: Precision score
        - recall: Recall score
        - roc_auc: Area under the Receiver Operating Characteristic curve
        - pr_auc: Area under the Precision-Recall curve
        - ap: Average Precision
        - confusion_matrix: Confusion matrix as a list of lists
        - auc_r: AUC-ROC (Area Under the ROC curve)
    """

    pred, pred_proba = pred.argmax(1), pred[:, 1]

    acc = ((pred == labels)).sum() / len(pred)

    true_pos = (np.where(pred == 1, 1, 0) + np.where(labels == 1, 1, 0) > 1).sum()
    false_pos = (np.where(pred == 1, 1, 0) + np.where(labels == 0, 1, 0) > 1).sum()
    false_neg = (np.where(pred == 0, 1, 0) + np.where(labels == 1, 1, 0) > 1).sum()
    true_neg = (np.where(pred == 0, 1, 0) + np.where(labels == 0, 1, 0) > 1).sum()

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0

    f_1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    confusion_matrix = pd.DataFrame(np.array([[true_pos, false_pos], [false_neg, true_neg]]),
                                    columns=["labels positive", "labels negative"],
                                    index=["predicted positive", "predicted negative"])

    average_precision = average_precision_score(labels, pred_proba)

    fpr, tpr, _ = roc_curve(labels, pred_proba)
    prc, rec, _ = precision_recall_curve(labels, pred_proba)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(rec, prc)
    auc_r = (fpr, tpr, roc_auc, name)
    return (acc, f_1, precision, recall, roc_auc, pr_auc, average_precision, confusion_matrix, auc_r)


def build_fsi_graph(train_data, col_drop):
    """Build a heterogeneous graph from an edgelist and node index.
    Parameters
    ----------
    train_data : cudf.DataFrame
        Training data containing node features.
    col_drop : list
        List of features to drop from the node features.
    Returns
    -------
    tuple
        A tuple containing the following elements:
        DGLGraph
            The built DGL graph representing the heterogeneous graph.
        torch.tensor
            Normalized feature tensor after dropping specified columns.
    Notes
    -----
    This function takes the training data, represented as a cudf DataFrame,
    and constructs a heterogeneous graph (DGLGraph) from the given edgelist
    and node index.

    The `col_drop` list specifies which features should be dropped from the
    node features to build the normalized feature tensor.

    Example
    -------
    >>> import cudf
    >>> train_data = cudf.DataFrame({'node_id': [1, 2, 3],
    ...                            'feature1': [0.1, 0.2, 0.3],
    ...                            'feature2': [0.4, 0.5, 0.6]})
    >>> col_drop = ['feature2']
    >>> graph, features = build_heterograph(train_data, col_drop)
    """

    feature_tensors = train_data.drop(col_drop, axis=1).values
    feature_tensors = torch.from_dlpack(feature_tensors.toDlpack())
    feature_tensors = (feature_tensors - feature_tensors.mean(0, keepdim=True)) / (0.0001 +
                                                                                   feature_tensors.std(0, keepdim=True))

    client_tensor, merchant_tensor, transaction_tensor = torch.tensor_split(
        torch.from_dlpack(train_data[col_drop].values.toDlpack()).long(), 3, dim=1)

    client_tensor, merchant_tensor, transaction_tensor = (client_tensor.view(-1),
                                                          merchant_tensor.view(-1),
                                                          transaction_tensor.view(-1))

    edge_list = {
        ('client', 'buy', 'transaction'): (client_tensor, transaction_tensor),
        ('transaction', 'bought', 'client'): (transaction_tensor, client_tensor),
        ('transaction', 'issued', 'merchant'): (transaction_tensor, merchant_tensor),
        ('merchant', 'sell', 'transaction'): (merchant_tensor, transaction_tensor)
    }

    graph = dgl.heterograph(edge_list)

    return graph, feature_tensors


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
    df_train = cf.read_csv(training_data)
    df_test = cf.read_csv(test_data)
    train_size = df_train.shape[0]
    cdf = cf.concat([df_train, df_test], axis=0)
    labels = cdf['fraud_label'].values
    cdf.drop(['fraud_label', 'index'], inplace=True, axis=1)

    cdf.reset_index(inplace=True)
    meta_cols = ['client_node', 'merchant_node']
    for col in meta_cols:
        cdf[col] = cf.CategoricalIndex(cdf[col]).codes

    return (cdf.iloc[:train_size, :],
            cdf.iloc[train_size:, :],
            cdf['index'][:train_size],
            cdf['index'][train_size:],
            labels,
            cdf)


def save_model(graph, model, hyperparameters, xgb_model, model_dir):
    """ Save the trained model and associated data to the specified directory.

    Parameters
    ----------
    graph : dgl.DGLGraph
        The graph object representing the data used for training the model.

    model : nn.Module
        The trained model object to be saved.

    hyperparameters : dict
        A dictionary containing the hyperparameters used for training the model.

    xgb_model : XGBoost
        The trained XGBoost model associated with the main model.

    model_dir : str
        The directory path where the model and associated data will be saved.

    """

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
    with open(os.path.join(model_dir, 'hyperparams.pkl'), 'wb') as f:
        pickle.dump(hyperparameters, f)
    with open(os.path.join(model_dir, 'graph.pkl'), 'wb') as f:
        pickle.dump(graph, f)
    xgb_model.save_model(os.path.join(model_dir, "xgb.pt"))


def load_model(model_dir, gnn_model=HinSAGE):
    """Load trained models from model directory

    Parameters
    ----------
    model_dir : str
        models directory path
    gnn_model: nn.Module
        GNN model type to load either HinSAGE or HeteroRGCN

    Returns
    -------
    (nn.Module, dgl.DGLHeteroGraph, dict)
        model, training graph, hyperparameter
    """

    with open(os.path.join(model_dir, "graph.pkl"), 'rb') as f:
        graph = pickle.load(f)
    with open(os.path.join(model_dir, 'hyperparams.pkl'), 'rb') as f:
        hyperparameters = pickle.load(f)
    model = gnn_model(graph,
                      in_size=hyperparameters['in_size'],
                      hidden_size=hyperparameters['hidden_size'],
                      out_size=hyperparameters['out_size'],
                      n_layers=hyperparameters['n_layers'],
                      embedding_size=hyperparameters['embedding_size'],
                      target=hyperparameters['target_node'])
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt')))
    xgb_model = ForestInference.load(os.path.join(model_dir, 'xgb.pt'), output_class=True)

    return model, xgb_model, graph


def init_loaders(g_train, train_idx, test_idx, val_idx, g_test, target_node='transaction', batch_size=100):
    """
    Initialize dataloader and graph sampler. For training, use neighbor sampling.

    Parameters
    ----------
    g_train : DGLHeteroGraph
        Train graph.
    train_idx : list
        Train feature index.
    test_idx : list
        Test feature index.
    val_idx : list
        Validation index.
    g_test : DGLHeteroGraph
        Test graph.
    target_node : str, optional
        Target node. Defaults to 'transaction'.
    batch_size : int
        Batch size for inference.
    Returns
    -------
    List[tuple]
        List of data loaders consisting of (DataLoader, DataLoader, DataLoader).


    Example
    -------
    >>> train_loader, test_loader, val_loader = initialize_dataloader(g_train, train_idx, test_idx,
            val_idx, g_test, target_node='authentication', batch_size=32)
    """

    neighbor_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    full_sampler = dgl.dataloading.MultiLayerNeighborSampler([4, 3])

    train_dataloader = dgl.dataloading.DataLoader(g_train, {target_node: train_idx},
                                                  neighbor_sampler,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=0,
                                                  use_uva=False)

    test_dataloader = dgl.dataloading.DataLoader(g_test, {target_node: test_idx},
                                                 full_sampler,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 num_workers=0,
                                                 use_uva=False)

    val_dataloader = dgl.dataloading.DataLoader(g_test, {target_node: val_idx},
                                                neighbor_sampler,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=0,
                                                use_uva=False)

    return train_dataloader, val_dataloader, test_dataloader


def train(model, loss_func, train_dataloader, labels, optimizer, feature_tensors, target_node='transaction'):
    """
    Train the specified GNN model using the given training data.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be trained.
    loss_func : callable
        The loss function to compute the training loss.
    train_dataloader : dgl.dataloading.DataLoader
        DataLoader containing the training dataset.
    labels : torch.Tensor
        The ground truth labels for the training data.
        Shape: (num_samples, num_classes).
    optimizer : torch.optim.Optimizer
        The optimizer used to update the model's parameters during training.
    feature_tensors : torch.Tensor
        The feature tensors corresponding to the training data.
        Shape: (num_samples, num_features).
    target_node : str, optional
        The target node for training, indicating the node of interest.
        Defaults to 'transaction'.

    Returns
    -------
    List[float, float]
        Training accuracy and training loss
    """

    model.train()
    train_loss = 0.0
    for _, (_, output_nodes, blocks) in enumerate(train_dataloader):
        seed = output_nodes[target_node]
        nid = blocks[0].srcnodes[target_node].data[dgl.NID]
        input_features = feature_tensors[nid]

        logits, _ = model(blocks, input_features)
        loss = loss_func(logits, labels[seed])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc = accuracy_score(logits.argmax(1).cpu(), labels[seed].cpu().long()).item()

    return train_acc, train_loss


@torch.no_grad()
def evaluate(model, eval_loader, feature_tensors, target_node):
    """Evaluate the specified model on the given evaluation input graph

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be evaluated.
    eval_loader : dgl.dataloading.DataLoader
        DataLoader containing the evaluation dataset.
    feature_tensors : torch.Tensor
        The feature tensors corresponding to the evaluation data.
        Shape: (num_samples, num_features).
    target_node : str
        The target node for evaluation, indicating the node of interest.

    Returns
    -------
    (torch.Tensor, torch.Tensor, torch.Tensor)
        A tuple containing numpy arrays of logits, eval seed and embeddings
    """
    model.eval()
    eval_logits = []
    eval_seeds = []
    embedding = []

    for _, output_nodes, blocks in eval_loader:

        seed = output_nodes[target_node]

        nid = blocks[0].srcnodes[target_node].data[dgl.NID]
        input_features = feature_tensors[nid]
        logits, embedd = model.infer(blocks, input_features)
        eval_logits.append(logits.cpu().detach())
        eval_seeds.append(seed)
        embedding.append(embedd)

    eval_logits = torch.cat(eval_logits)
    eval_seeds = torch.cat(eval_seeds)
    embedding = torch.cat(embedding)
    return eval_logits, eval_seeds, embedding


@click.command()
@click.option('--training-data', help="Path to training data ", default="data/training.csv")
@click.option('--validation-data', help="Path to validation data", default="data/validation.csv")
@click.option('--model-dir', help="path to model directory", default="debug")
@click.option('--target-node', help="Target node", default="transaction")
@click.option('--epochs', help="Number of epochs", default=20)
@click.option('--batch_size', help="Batch size", default=1024)
@click.option('--output-file', help="Path to csv inference result", default="debug/out.csv")
@click.option('--model-type', help="Model type either RGCN/Graphsage", default="RGCN")
def train_model(training_data, validation_data, model_dir, target_node, epochs, batch_size, output_file, model_type):
    from timeit import default_timer as timer
    start = timer()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model_type == "RGCN":
        gnn_model = HeteroRGCN
    else:
        gnn_model = HinSAGE

    # process training data
    train_data, _, train_idx, inductive_idx, labels, df = prepare_data(training_data, validation_data)

    meta_cols = ["client_node", "merchant_node", "index"]

    # Build graph
    whole_graph, feature_tensors = build_fsi_graph(df, meta_cols)
    train_graph, _ = build_fsi_graph(train_data, meta_cols)

    feature_tensors = feature_tensors.float()
    train_idx = torch.from_dlpack(train_idx.values.toDlpack()).long()
    inductive_idx = torch.from_dlpack(inductive_idx.values.toDlpack()).long()
    labels = torch.from_dlpack(labels.toDlpack()).long()

    # Hyperparameters
    in_size, hidden_size, out_size, n_layers, embedding_size = 111, 64, 2, 2, 1
    hyperparameters = {
        "in_size": in_size,
        "hidden_size": hidden_size,
        "out_size": out_size,
        "n_layers": n_layers,
        "embedding_size": embedding_size,
        "target_node": target_node,
        "epoch": epochs
    }

    scale_pos_weight = (labels[train_idx].sum() / train_data.shape[0]).item()
    scale_pos_weight = torch.FloatTensor([scale_pos_weight, 1 - scale_pos_weight]).to(device)

    # Dataloaders
    train_loader, val_loader, test_loader = init_loaders(train_graph, train_idx, test_idx=inductive_idx,
                                                         val_idx=inductive_idx, g_test=whole_graph,
                                                         batch_size=batch_size)

    # Set model variables
    model = gnn_model(whole_graph, in_size, hidden_size, out_size, n_layers, embedding_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_func = nn.CrossEntropyLoss(weight=scale_pos_weight.float())

    for epoch in trange(epochs):

        train_acc, loss = train(model, loss_func, train_loader, labels, optimizer, feature_tensors, target_node)
        print(f"Epoch {epoch}/{epochs} | Train Accuracy: {train_acc} | Train Loss: {loss}")
        val_logits, val_seed, _ = evaluate(model, val_loader, feature_tensors, target_node)
        val_accuracy = accuracy(val_logits.argmax(1), labels.long()[val_seed].cpu(), "binary").item()
        val_auc = roc_auc_score(
            labels.long()[val_seed].cpu().numpy(),
            val_logits[:, 1].numpy(),
        )
        print(f"Validation Accuracy: {val_accuracy} auc {val_auc}")

    # Create embeddings
    _, train_seeds, train_embedding = evaluate(model, train_loader, feature_tensors, target_node)
    test_logits, test_seeds, test_embedding = evaluate(model, test_loader, feature_tensors, target_node)

    # compute metrics
    test_acc = accuracy(test_logits.argmax(dim=1), labels.long()[test_seeds].cpu(), "binary").item()
    test_auc = roc_auc_score(labels.long()[test_seeds].cpu().numpy(), test_logits[:, 1].numpy())

    metrics_result = pd.DataFrame()
    print(f"Final Test Accuracy: {test_acc} auc {test_auc}")
    acc, f_1, precision, recall, roc_auc, pr_auc, average_precision, _, _ = get_metrics(
        test_logits.numpy(), labels[test_seeds].cpu().numpy())
    metrics_result = [{
        'model': 'RGCN',
        'acc': acc,
        'f1': f_1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'ap': average_precision
    }]

    # Train XGBoost classifier on embedding vector
    classifier = XGBClassifier(n_estimators=100)
    classifier.fit(train_embedding.cpu().numpy(), labels[train_seeds].cpu().numpy())
    xgb_pred = classifier.predict_proba(test_embedding.cpu().numpy())
    acc, f_1, precision, recall, roc_auc, pr_auc, average_precision, _, _ = get_metrics(
        xgb_pred, labels[inductive_idx].cpu().numpy(),  name='RGCN_XGB')
    metrics_result += [{
        'model': 'RGCN_XGB',
        'acc': acc,
        'f1': f_1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'ap': average_precision
    }]

    # Save model
    pd.DataFrame(metrics_result).to_csv(output_file)
    save_model(whole_graph, model, hyperparameters, classifier, model_dir)

    end = timer()
    print(end - start)


if __name__ == "__main__":

    train_model()
