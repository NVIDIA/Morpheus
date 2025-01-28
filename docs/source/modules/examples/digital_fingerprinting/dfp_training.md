<!--
SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# DFP Training Module

This module function is responsible for training the model.

## Configurable Parameters

| Parameter       | Type  | Description                            | Example Value                                                                                                                                                                                                                                                 | Default Value |
|-----------------|-------|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `feature_columns` | `list`  | List of feature columns to train on    | `["column1", "column2", "column3"]` | `-`           |
| `epochs`          | `int`   | Number of epochs to train for          | `50` | `-`           |
| `model_kwargs`    | `dict`  | Keyword arguments to pass to the model | `{"encoder_layers": [64, 32], "decoder_layers": [32, 64], "activation": "relu", "swap_p": 0.1, "lr": 0.001, "lr_decay": 0.9, "batch_size": 32, "verbose": 1, "optimizer": "adam", "scalar": "min_max", "min_cats": 10, "progress_bar": false, "device": "cpu"}` | `-`             |
| `validation_size` | `float` | Size of the validation set             | `0.1 ` | `-`           |

## JSON Example

```json
{
  "feature_columns": [
    "column1",
    "column2",
    "column3"
  ],
  "epochs": 50,
  "model_kwargs": {
    "encoder_layers": [
      64,
      32
    ],
    "decoder_layers": [
      32,
      64
    ],
    "activation": "relu",
    "swap_p": 0.1,
    "lr": 0.001,
    "lr_decay": 0.9,
    "batch_size": 32,
    "verbose": 1,
    "optimizer": "adam",
    "scalar": "min_max",
    "min_cats": 10,
    "progress_bar": false,
    "device": "cpu"
  },
  "validation_size": 0.1
}
```
