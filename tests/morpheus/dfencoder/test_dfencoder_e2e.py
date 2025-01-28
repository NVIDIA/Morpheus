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

import numpy as np
import pandas as pd
import pytest

from _utils import TEST_DIRS
from morpheus.models.dfencoder.autoencoder import AutoEncoder

INFERENCE_START_DATE = "2022-11-01"
VALIDATION_SET_SIZE = 3000  # around 1/10 of the train set
TRAIN_VAL_SPLIT_SEED = 11

FEATURE_COLUMNS = [
    "app_name",
    "browser_type",
    "os",
    "has_error",
    "country",
    "city",
    "log_count",
    "location_incr",
    "app_incr",
]
LOSS_TYPES = ["train", "val", "id_val"]
# 75th quantile of the losses from 100 times of offline training
LOSS_TARGETS = {
    "train": {
        "log_count": 0.33991,
        "location_incr": 0.30789,
        "app_incr": 0.17698,
        "has_error": 0.00878,
        "app_name": 0.13066,
        "browser_type": 0.39804,
        "os": 0.09882,
        "country": 0.06063,
        "city": 0.32344,
    },
    "val": {
        "log_count": 0.3384,
        "location_incr": 0.31456,
        "app_incr": 0.16201,
        "has_error": 0.00614,
        "app_name": 0.11907,
        "browser_type": 0.38239,
        "os": 0.00064,
        "country": 0.0042,
        "city": 0.32161,
    },
    "id_val": {
        "log_count": 0.07079,
        "location_incr": 0.05318,
        "app_incr": 0.03659,
        "has_error": 0.0046,
        "app_name": 0.03542,
        "browser_type": 0.0915,
        "os": 0.00057,
        "country": 0.00343,
        "city": 0.08525,
    },
}
LOSS_TOLERANCE_RATIO = 1.25


@pytest.mark.slow
@pytest.mark.usefixtures("manual_seed")
def test_dfencoder_e2e():
    # Load data
    input_file = os.path.join(TEST_DIRS.validation_data_dir, "dfp_training_feature_samples.csv")
    data = pd.read_csv(input_file, index_col=0)
    train_and_validation_df = data[data.date < INFERENCE_START_DATE].sample(frac=1, random_state=TRAIN_VAL_SPLIT_SEED)
    inference_df = data[data.date >= INFERENCE_START_DATE][FEATURE_COLUMNS]
    train_df, validation_df = (
        train_and_validation_df[VALIDATION_SET_SIZE:][FEATURE_COLUMNS],
        train_and_validation_df[:VALIDATION_SET_SIZE][FEATURE_COLUMNS],
    )

    # Training
    model = AutoEncoder(
        encoder_layers=[512, 500],  # layers of the encoding part
        decoder_layers=[512],  # layers of the decoding part
        activation="relu",  # activation function
        swap_probability=0.2,  # noise parameter
        learning_rate=0.01,  # learning rate
        learning_rate_decay=0.99,  # learning decay
        batch_size=1024,
        logger="basic",
        verbose=False,
        progress_bar=False,
        optimizer="adam",  # SGD optimizer is selected(Stochastic gradient descent)
        scaler="standard",  # feature scaling method
        min_cats=1,  # cut off for minority categories
        patience=5,
        distributed_training=False,
    )
    model.fit(
        train_df,
        epochs=10,
        validation_data=validation_df,
        run_validation=True,
        use_val_for_loss_stats=True,
    )

    # Make sure model converges (low loss)
    for loss_type in LOSS_TYPES:
        ft_losses = getattr(model.logger, f"{loss_type}_fts")
        for feature, losses_l in ft_losses.items():
            losses = losses_l[1]
            assert min(losses) < LOSS_TARGETS[loss_type][feature] * LOSS_TOLERANCE_RATIO

    # Inference
    inf_res = model.get_results(inference_df)

    # Assert the consistency of output rows and columns
    assert len(inf_res) == len(inference_df)
    assert sorted(inf_res.columns) == sorted(
        [feature + col_suffix for feature in FEATURE_COLUMNS
         for col_suffix in ["", "_pred", "_loss", "_z_loss"]] + ["max_abs_z", "mean_abs_z", "z_loss_scaler_type"])
    # make sure the user baseline is modeled well enough so the minimum and median z scores
    # from inference are in range
    assert min(inf_res.mean_abs_z) < 1

    # expect median mean_abs_z to be < 50. Using 100 to leave some room for variability
    assert (np.median(inf_res.mean_abs_z) < 100)
