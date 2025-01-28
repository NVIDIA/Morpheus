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

import os
import sys

import pytest

from _utils import TEST_DIRS
from _utils import import_or_skip
from _utils import remove_module

SKIP_REASON = ("Tests for the gnn_fraud_detection_pipeline example require a number of packages not installed in the "
               "Morpheus development environment. See `examples/gnn_fraud_detection_pipeline/README.md` for details on "
               "installing these additional dependencies")


@pytest.fixture(name="dgl", autouse=True, scope='session')
def dgl_fixture(fail_missing: bool):
    """
    All of the tests in this subdir require dgl
    """
    yield import_or_skip("dgl", reason=SKIP_REASON, fail_missing=fail_missing)


@pytest.fixture(name="cuml", autouse=True, scope='session')
def cuml_fixture(fail_missing: bool):
    """
    All of the tests in this subdir require cuml
    """
    yield import_or_skip("cuml", reason=SKIP_REASON, fail_missing=fail_missing)


@pytest.fixture(name="config")
def config_fixture(config):
    """
    The GNN fraud detection pipeline utilizes the "other" pipeline mode.
    """
    from morpheus.config import PipelineModes
    config.mode = PipelineModes.OTHER
    yield config


@pytest.fixture(name="manual_seed", scope="function", autouse=True)
def manual_seed_fixture(manual_seed):
    """
    Extends the base `manual_seed` fixture to also set the seed for dgl, ensuring deterministic results in tests
    """
    import dgl

    def seed_fn(seed=42):
        manual_seed(seed)
        dgl.seed(seed)

    seed_fn()
    yield seed_fn


@pytest.fixture(scope='session', name="example_dir")
def example_dir_fixture():
    yield os.path.join(TEST_DIRS.examples_dir, 'gnn_fraud_detection_pipeline')


@pytest.fixture(scope='session', name="training_file")
def training_file_fixture(example_dir: str):
    yield os.path.join(example_dir, 'training.csv')


@pytest.fixture(scope='session', name="model_dir")
def model_dir_fixture(example_dir: str):
    yield os.path.join(example_dir, 'model')


@pytest.fixture(scope='session', name="xgb_model")
def xgb_model_fixture(model_dir: str):
    yield os.path.join(model_dir, 'xgb.pt')


# Some of the code inside gnn_fraud_detection_pipeline performs some relative imports in the form of:
#    from .mod import Class
# For this reason we need to ensure that the examples dir is in the sys.path first
@pytest.mark.usefixtures("restore_sys_path", "reset_plugins")
@pytest.fixture(name="ex_in_sys_path", autouse=True)
def ex_in_sys_path_fixture(example_dir: str):
    sys.path.insert(0, example_dir)


@pytest.fixture(autouse=True)
def reset_modules():
    """
    Other examples have a stages module, ensure it is un-imported after running tests in this subdir
    """
    yield
    remove_module('stages')


@pytest.fixture(name="test_data")
def test_data_fixture():
    """
    Construct test data, a small DF of 10 rows which we will build a graph from
    The nodes in our graph will be the unique values from each of our three columns, and the index is also
    representing our transaction ids.
    There is only one duplicated value (2697) in our dataset so we should expect 29 nodes
    Our expected edges will be each value in client_node and merchant_node to their associated index value ex:
    (795, 2) & (8567, 2)
    thus we should expect 20 edges, although 2697 is duplicated in the client_node column we should expect two
    unique edges for each entry (2697, 14) & (2697, 91)
    """
    import cudf
    index = [2, 14, 16, 26, 41, 42, 70, 91, 93, 95]

    client_data = [795.0, 2697.0, 5531.0, 415.0, 2580.0, 3551.0, 6547.0, 2697.0, 3503.0, 7173.0]
    merchant_data = [8567.0, 4609.0, 2781.0, 7844.0, 629.0, 6915.0, 7071.0, 570.0, 2446.0, 8110.0]

    df_data = {
        'index': index,
        'client_node': client_data,
        'merchant_node': merchant_data,
        'fraud_label': [1 for _ in range(len(index))]
    }

    # Fill in the other columns so that we match the shape the model is expecting
    for i in range(1000, 1113):
        # these two values are skipped, apparently place-holders for client_node & merchant_node
        if i not in (1002, 1003):
            df_data[str(i)] = [0.0 for _ in range(len(index))]

    df = cudf.DataFrame(df_data, index=index)

    # Create indexed nodeId
    meta_cols = ['index', 'client_node', 'merchant_node']
    for col in meta_cols:
        df[col] = cudf.CategoricalIndex(df[col]).codes
    df.index = df['index']

    # Collect expected nodes, since hetero nodes could share same index
    # We use dict of node_name:index
    expected_nodes = {}
    for col in meta_cols:
        expected_nodes[col] = set(df[col].to_arrow().tolist())

    # ensuring test data & assumptions are correct
    assert sum(len(nodes) for _, nodes in expected_nodes.items()) == 29

    expected_edges = {'buy': [], 'sell': []}
    for i in range(df.shape[0]):
        for key, val in {'buy': 'client_node', 'sell': 'merchant_node'}.items():
            expected_edges[key].append([df[val].iloc[i], i])

    # ensuring test data & assumptions are correct
    assert sum(len(edges) for _, edges in expected_edges.items()) == 20

    yield {
        "index": df['index'].to_arrow().tolist(),
        "client_data": df['client_node'].to_arrow().tolist(),
        "merchant_data": df['merchant_node'].to_arrow().tolist(),
        "df": df,
        "expected_nodes": expected_nodes,
        "expected_edges": expected_edges
    }
