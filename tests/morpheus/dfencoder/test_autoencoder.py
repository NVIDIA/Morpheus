#!/usr/bin/env python
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

import os
import typing
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset as TorchDataset

from _utils import TEST_DIRS
from _utils.dataset_manager import DatasetManager
from morpheus.config import AEFeatureScalar
from morpheus.models.dfencoder import ae_module
from morpheus.models.dfencoder import autoencoder
from morpheus.models.dfencoder import scalers
from morpheus.models.dfencoder.dataframe import EncoderDataFrame
from morpheus.models.dfencoder.dataloader import DataframeDataset
from morpheus.models.dfencoder.dataloader import DFEncoderDataLoader
from morpheus.models.dfencoder.dataloader import FileSystemDataset

# Only pandas and Python is supported
pytestmark = [pytest.mark.use_pandas, pytest.mark.cpu_mode]

BIN_COLS = ['ts_anomaly']

CAT_COLS = [
    'apiVersion',
    'errorCode',
    'errorMessage',
    'eventName',
    'eventSource',
    'eventTime',
    'sourceIPAddress',
    'userAgent',
    'userIdentityaccessKeyId',
    'userIdentityaccountId',
    'userIdentityarn',
    'userIdentityprincipalId',
    'userIdentitysessionContextsessionIssueruserName',
    'userIdentitytype'
]

NUMERIC_COLS = ['eventID', 'ae_anomaly_score']


@pytest.fixture(name="train_ae", scope="function")
def train_ae_fixture():
    """
    Construct an AutoEncoder instance with the same values used by `train_ae_stage`
    """
    yield autoencoder.AutoEncoder(
        encoder_layers=[512, 500],
        decoder_layers=[512],
        activation='relu',
        swap_probability=0.2,
        learning_rate=0.01,
        learning_rate_decay=.99,
        batch_size=512,
        verbose=False,
        optimizer='sgd',
        scaler='standard',
        min_cats=1,
        progress_bar=False,
    )


@pytest.fixture(name="train_df", scope="function")
def train_df_fixture(dataset_pandas: DatasetManager) -> typing.Iterator[pd.DataFrame]:
    yield dataset_pandas[os.path.join(TEST_DIRS.validation_data_dir, "dfp-cloudtrail-role-g-validation-data-input.csv")]


@pytest.fixture(name="expected_device_type", scope="session")
def expected_device_type_fixture() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def compare_numeric_features(features, expected_features):
    assert sorted(features.keys()) == sorted(expected_features.keys())
    for (feature, expected_vals) in expected_features.items():
        ae_vals = features[feature]
        assert round(ae_vals['mean'], 2) == expected_vals['mean'], \
            f"Mean value of feature:{feature} does not match {round(ae_vals['mean'], 2)}!= {expected_vals['mean']}"

        assert round(ae_vals['std'], 2) == expected_vals['std'], \
            f"Mean value of feature:{feature} does not match {round(ae_vals['std'], 2)}!= {expected_vals['std']}"

        assert isinstance(ae_vals['scaler'], expected_vals['scaler_cls'])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA enabled torch installation")
def test_ohe():
    tensor = torch.tensor(range(4), dtype=torch.int64)
    results = autoencoder._ohe(tensor, 4, device="cpu")
    expected = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    assert results.device.type == "cpu"
    assert torch.equal(results, expected), f"{results} != {expected}"

    results = autoencoder._ohe(tensor.to("cuda", copy=True), 4, device="cuda")
    assert results.device.type == "cuda"
    assert torch.equal(results, expected.to("cuda", copy=True)), f"{results} != {expected}"


@pytest.mark.parametrize("num_cats,expected", [(0, 0), (5, 4), (20, 9), (40000, 600)])
def test_compute_embedding_size(num_cats: int, expected: int):
    assert ae_module._compute_embedding_size(num_cats) == expected


def test_complete_layer_constructor():
    complete_layer = ae_module.CompleteLayer(4, 5)
    assert len(complete_layer.layers) == 1
    assert isinstance(complete_layer.layers[0], torch.nn.Linear)
    assert complete_layer.layers[0].in_features == 4
    assert complete_layer.layers[0].out_features == 5

    complete_layer = ae_module.CompleteLayer(4, 5, activation='tanh')
    assert len(complete_layer.layers) == 2
    assert complete_layer.layers[1] is torch.tanh

    complete_layer = ae_module.CompleteLayer(4, 5, dropout=0.2)
    assert len(complete_layer.layers) == 2
    assert isinstance(complete_layer.layers[1], torch.nn.Dropout)
    assert complete_layer.layers[1].p == 0.2

    complete_layer = ae_module.CompleteLayer(6, 11, activation='sigmoid', dropout=0.3)
    assert len(complete_layer.layers) == 3
    assert isinstance(complete_layer.layers[0], torch.nn.Linear)
    assert complete_layer.layers[0].in_features == 6
    assert complete_layer.layers[0].out_features == 11
    assert complete_layer.layers[1] is torch.sigmoid
    assert isinstance(complete_layer.layers[2], torch.nn.Dropout)
    assert complete_layer.layers[2].p == 0.3


def test_complete_layer_interpret_activation():
    complete_layer = ae_module.CompleteLayer(4, 5)
    assert complete_layer.interpret_activation('elu') is torch.nn.functional.elu

    # Test for bad activation, this really does raise the base Exception class.
    with pytest.raises(Exception):
        complete_layer.interpret_activation()

    with pytest.raises(Exception):
        complete_layer.interpret_activation("does_not_exist")

    complete_layer = ae_module.CompleteLayer(6, 11, activation='sigmoid')
    assert complete_layer.interpret_activation() is torch.sigmoid


@pytest.mark.usefixtures("manual_seed")
def test_complete_layer_forward():
    # Setting dropout probability to 0. The results of dropout our deterministic, but are only
    # consistent when run on the same GPU.
    complete_layer = ae_module.CompleteLayer(3, 5, activation='tanh', dropout=0)
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.float32)
    results = complete_layer.forward(tensor)
    expected = torch.tensor([[0.7223, 0.7902, 0.9647, 0.5613, 0.9163], [0.9971, 0.9897, 0.9988, 0.8317, 0.9992],
                             [1.0000, 0.9995, 1.0000, 0.9417, 1.0000], [1.0000, 1.0000, 1.0000, 0.9806, 1.0000]],
                            dtype=torch.float32)

    assert torch.equal(torch.round(results, decimals=4), expected), f"{results} != {expected}"


def test_auto_encoder_constructor_default_vals(expected_device_type: str):
    ae = autoencoder.AutoEncoder()
    assert isinstance(ae.model, torch.nn.Module)
    assert ae.model.encoder_layers is None
    assert ae.model.decoder_layers is None
    assert ae.min_cats == 10
    assert ae.swap_probability == 0.15
    assert ae.batch_size == 256
    assert ae.eval_batch_size == 1024
    assert ae.model.activation == 'relu'
    assert ae.optimizer == 'adam'
    assert ae.learning_rate == 0.01
    assert ae.learning_rate_decay is None
    assert ae.device.type == expected_device_type
    assert ae.scaler == 'standard'
    assert ae.loss_scaler is scalers.StandardScaler


def test_auto_encoder_constructor(train_ae: autoencoder.AutoEncoder, expected_device_type: str):
    """
    Test copnstructor invokation using the values used by `train_ae_stage`
    """
    assert isinstance(train_ae.model, torch.nn.Module)
    assert train_ae.model.encoder_layers == [512, 500]
    assert train_ae.model.decoder_layers == [512]
    assert train_ae.min_cats == 1
    assert train_ae.swap_probability == 0.2
    assert train_ae.batch_size == 512
    assert train_ae.eval_batch_size == 1024
    assert train_ae.model.activation == 'relu'
    assert train_ae.optimizer == 'sgd'
    assert train_ae.learning_rate == 0.01
    assert train_ae.learning_rate_decay == 0.99
    assert not train_ae.progress_bar
    assert not train_ae.verbose
    assert train_ae.device.type == expected_device_type
    assert train_ae.scaler == 'standard'
    assert train_ae.loss_scaler is scalers.StandardScaler


def test_auto_encoder_get_scaler():
    ae = autoencoder.AutoEncoder()

    # Test the values in the `AEFeatureScalar` enum
    test_values = [(AEFeatureScalar.NONE, scalers.NullScaler), (AEFeatureScalar.STANDARD, scalers.StandardScaler),
                   (AEFeatureScalar.GAUSSRANK, scalers.GaussRankScaler)]

    assert len(test_values) == len(AEFeatureScalar), "Not all values in AEFeatureScalar are tested"

    for (value, expected) in test_values:
        assert ae.get_scaler(value) is expected


def test_auto_encoder_init_numeric(filter_probs_df):
    ae = autoencoder.AutoEncoder()
    ae._init_numeric(filter_probs_df)

    expected_features = {
        'v1': {
            'mean': 0.46, 'std': 0.35, 'scaler_cls': scalers.StandardScaler
        },
        'v2': {
            'mean': 0.51, 'std': 0.31, 'scaler_cls': scalers.StandardScaler
        },
        'v3': {
            'mean': 0.46, 'std': 0.3, 'scaler_cls': scalers.StandardScaler
        },
        'v4': {
            'mean': 0.54, 'std': 0.27, 'scaler_cls': scalers.StandardScaler
        }
    }

    # AE stores the features in an OrderedDict, but we don't want to be dependent on the order that Pandas reads in the
    # columns of a dataframe.
    assert sorted(ae.num_names) == sorted(expected_features.keys())
    compare_numeric_features(ae.numeric_fts, expected_features)


@pytest.mark.parametrize("input_type", [pd.DataFrame, FileSystemDataset, DFEncoderDataLoader, TorchDataset])
def test_auto_encoder_fit(train_ae: autoencoder.AutoEncoder, train_df: pd.DataFrame, input_type):
    _train_df = train_df
    if (isinstance(input_type, FileSystemDataset)):
        _train_df = DataframeDataset(_train_df)
    elif (isinstance(input_type, DFEncoderDataLoader)):
        _train_df = DFEncoderDataLoader.get_distributed_training_dataloader_from_df(train_ae,
                                                                                    _train_df,
                                                                                    1,
                                                                                    1,
                                                                                    num_workers=1)
    elif (isinstance(input_type, TorchDataset)):
        with pytest.raises(TypeError):
            train_ae.fit(_train_df, epochs=1)
        return

    train_ae.fit(_train_df, epochs=1)

    expected_numeric_features = {
        'eventID': {
            'mean': 156.5, 'std': 90.79, 'scaler_cls': scalers.StandardScaler
        },
        'ae_anomaly_score': {
            'mean': 1.67, 'std': 0.38, 'scaler_cls': scalers.StandardScaler
        }
    }

    assert sorted(train_ae.num_names) == sorted(NUMERIC_COLS)
    compare_numeric_features(train_ae.numeric_fts, expected_numeric_features)

    expected_bin_features = {'ts_anomaly': {'cats': [True, False], True: True, False: False}}
    assert train_ae.bin_names == BIN_COLS
    assert train_ae.binary_fts == expected_bin_features

    assert sorted(train_ae.categorical_fts.keys()) == CAT_COLS
    for cat in CAT_COLS:
        assert sorted(train_ae.categorical_fts[cat]['cats']) == sorted(_train_df[cat].dropna().unique())

    assert len(train_ae.cyclical_fts) == 0

    all_feature_names = sorted(NUMERIC_COLS + BIN_COLS + CAT_COLS)

    assert sorted(train_ae.feature_loss_stats.keys()) == all_feature_names
    for feature in train_ae.feature_loss_stats.values():
        assert isinstance(feature['scaler'], scalers.StandardScaler)

    assert isinstance(train_ae.optim, torch.optim.SGD)
    assert isinstance(train_ae.learning_rate_decay, torch.optim.lr_scheduler.ExponentialLR)
    assert train_ae.learning_rate_decay.gamma == 0.99
    assert train_ae.optim is train_ae.learning_rate_decay.optimizer


def test_auto_encoder_fit_early_stopping(train_df: pd.DataFrame):
    train_data = train_df.sample(frac=0.7, random_state=1)
    validation_data = train_df.drop(train_data.index)

    epochs = 10

    # Test normal training loop with no early stopping
    ae = autoencoder.AutoEncoder(patience=5)
    ae.fit(train_data, validation_data=validation_data, run_validation=True, use_val_for_loss_stats=True, epochs=epochs)
    # assert that training runs through all epoches
    assert ae.logger.n_epochs == epochs

    class MockHelper:
        """A helper class for mocking the `_validate_dataframe` method in the `AutoEncoder` class."""

        def __init__(self, orig_losses):
            """
            Initialization.

            Parameters:
            -----------
            mean_loss: list
                A list of mean validation losses to be returned by the mocked `_validate_dataframe` method.
            """
            self.orig_losses = orig_losses
            # counter to keep track of the number of times the mocked `_validate_dataframe` method has been called
            self.count = 0

        def mocked_validate_dataset(self, *args, **kwargs):  # pylint: disable=unused-argument
            """
            A mocked version of the `_validate_dataframe` method in the `AutoEncoder` class for testing early stopping.

            Returns:
            --------
            tuple of (float, float)
                A tuple of original validation loss and swapped loss values for each epoch.
            """
            mean_loss = self.orig_losses[self.count]
            self.count += 1
            return mean_loss

    # Test early stopping
    orig_losses = [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    ae = autoencoder.AutoEncoder(
        patience=3)  # should stop at epoch 3 as the first 3 losses are monotonically increasing
    mock_helper = MockHelper(orig_losses=orig_losses)  # validation loss is strictly increasing
    with patch.object(ae, '_validate_dataset', side_effect=mock_helper.mocked_validate_dataset):
        ae.fit(train_data,
               validation_data=validation_data,
               run_validation=True,
               use_val_for_loss_stats=True,
               epochs=epochs)
        # assert that training early-stops at epoch 3
        assert ae.logger.n_epochs == 3

    ae = autoencoder.AutoEncoder(
        patience=5)  # should stop at epoch 9 as losses from epoch 5-9 are monotonically increasing
    mock_helper = MockHelper(orig_losses=orig_losses)  # validation loss is strictly increasing
    with patch.object(ae, '_validate_dataset', side_effect=mock_helper.mocked_validate_dataset):
        ae.fit(train_data,
               validation_data=validation_data,
               run_validation=True,
               use_val_for_loss_stats=True,
               epochs=epochs)
        # assert that training early-stops at epoch 3
        assert ae.logger.n_epochs == 9


@pytest.mark.usefixtures("manual_seed")
def test_auto_encoder_get_anomaly_score(train_ae: autoencoder.AutoEncoder, train_df: pd.DataFrame):
    train_ae.fit(train_df, epochs=1)
    anomaly_score = train_ae.get_anomaly_score(train_df)
    assert len(anomaly_score) == len(train_df)
    assert round(anomaly_score.mean().item(), 2) == 2.29
    assert round(anomaly_score.std().item(), 2) == 0.11


def test_auto_encoder_get_anomaly_score_losses(train_ae: autoencoder.AutoEncoder):
    # create a dummy DataFrame with numerical and boolean features only
    row_cnt = 10
    # create a dummy DataFrame with categorical features
    data = {
        'num_1': list(range(row_cnt)),
        'num_2': [i / 2 for i in range(row_cnt)],
        'num_3': [i / 2 for i in range(row_cnt)],
        'bool_1': [i % 2 == 0 for i in range(row_cnt)],
        'bool_2': [i % 3 == 0 for i in range(row_cnt)],
        'cat_1': [f'str_{i}' for i in range(row_cnt)]
    }
    df = pd.DataFrame(data)

    train_ae._build_model(df)

    # call the function and check the output
    mse_loss, bce_loss, cce_loss = train_ae.get_anomaly_score_losses(df)

    # check that the output is of the correct shape
    assert mse_loss.shape == torch.Size([row_cnt, 3]), "mse_loss has incorrect shape"
    assert bce_loss.shape == torch.Size([row_cnt, 2]), "bce_loss has incorrect shape"
    assert cce_loss.shape == torch.Size([row_cnt, 1]), "cce_loss has incorrect shape"


def test_auto_encoder_get_anomaly_score_losses_no_cat_feats(train_ae: autoencoder.AutoEncoder):
    # create a dummy DataFrame with numerical and boolean features only
    row_cnt = 10
    data = {
        'num_1': list(range(row_cnt)),
        'bool_1': [i % 2 == 0 for i in range(row_cnt)],
        'bool_2': [i % 3 == 0 for i in range(row_cnt)]
    }
    df = pd.DataFrame(data)

    train_ae._build_model(df)

    # call the function and check the output
    mse_loss, bce_loss, cce_loss = train_ae.get_anomaly_score_losses(df)

    # check that the output is of the correct shape
    assert mse_loss.shape == torch.Size([row_cnt, 1]), "mse_loss has incorrect shape"
    assert bce_loss.shape == torch.Size([row_cnt, 2]), "bce_loss has incorrect shape"
    assert cce_loss.shape == torch.Size([row_cnt, 0]), "cce_loss has incorrect shape"


def test_auto_encoder_prepare_df(train_ae: autoencoder.AutoEncoder, train_df: pd.DataFrame):
    train_ae.fit(train_df, epochs=1)

    dfc = train_df.copy(deep=True)
    prepared_df = train_ae.prepare_df(dfc)

    assert isinstance(prepared_df, EncoderDataFrame)

    for feature in NUMERIC_COLS:
        scaler = scalers.StandardScaler()
        scaler.fit(train_df[feature].values)
        expected_values = scaler.transform(train_df[feature].values.copy())

        assert (prepared_df[feature].values == expected_values).all(), \
            f"Values for feature {feature} do not match {prepared_df[feature]} != {expected_values}"

    # Bin features should remain the same when the input is already boolean, this DF only has one
    assert (prepared_df.ts_anomaly == train_df.ts_anomaly).all()

    for cat in CAT_COLS:
        assert cat in prepared_df
        isinstance(prepared_df[cat], pd.Categorical)
        assert not prepared_df[cat].hasnans

        if train_df[cat].hasnans:
            assert '_other' in prepared_df[cat].values
        else:
            assert '_other' not in prepared_df[cat].values


def test_build_input_tensor(train_ae: autoencoder.AutoEncoder, train_df: pd.DataFrame, expected_device_type: str):
    train_ae.fit(train_df, epochs=1)
    prepared_df = train_ae.prepare_df(train_df)
    tensor = train_ae.build_input_tensor(prepared_df)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.device.type == expected_device_type
    assert len(tensor) == len(train_df)


@pytest.mark.usefixtures("manual_seed")
def test_auto_encoder_get_results(train_ae: autoencoder.AutoEncoder, train_df: pd.DataFrame):
    train_ae.fit(train_df, epochs=1)
    results = train_ae.get_results(train_df)

    for feature in sorted(NUMERIC_COLS + BIN_COLS + CAT_COLS):
        assert feature in results.columns
        assert f'{feature}_pred' in results.columns
        assert f'{feature}_loss' in results.columns
        assert f'{feature}_z_loss' in results.columns

    assert 'max_abs_z' in results.columns
    assert 'mean_abs_z' in results.columns

    assert np.isclose(results.loc[0, 'max_abs_z'], 2.51, atol=1e-2)

    # Numpy float has different precision checks than python float, so we wrap it.
    assert np.isclose(results.loc[0, 'mean_abs_z'], 0.361, atol=1e-3)
    assert results.loc[0, 'z_loss_scaler_type'] == 'z'


@pytest.mark.usefixtures("manual_seed")
def test_auto_encoder_num_only_convergence(train_ae: autoencoder.AutoEncoder):
    num_df = pd.DataFrame({
        'num_feat_1': [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9],
        'num_feat_2': [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1],
    })

    with warnings.catch_warnings():
        # Ignore warning regarding tensorflow not being installed
        warnings.filterwarnings("ignore", message="Initializing zero-element tensors is a no-op", category=UserWarning)
        train_ae.fit(num_df, epochs=50)

    avg_loss = np.sum([np.array(loss[1])
                       for loss in train_ae.logger.train_fts.values()], axis=0) / len(train_ae.logger.train_fts)

    # Make sure the model converges with numerical feats only
    assert avg_loss[-1] < avg_loss[0] * 0.75
