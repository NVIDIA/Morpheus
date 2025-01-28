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

## DFP Training Pipe Module

This module function consolidates multiple DFP pipeline modules relevant to the training process into a single module.

### Configurable Parameters

| Key                          | Type | Description                                                                             | Example Value | Default Value |
|------------------------------|------|-----------------------------------------------------------------------------------------|---------------|---------------|
| `timestamp_column_name`      | `str`  | Name of the timestamp column used in the data.                                          | `"timestamp"`   | `-`           |
| `cache_dir`                  | `str`  | Directory to cache the rolling window data.                                             | `"/tmp/cache"`  | `-`           |
| `batching_options`           | `dict` | Options for batching files.                                                             | Refer Below    | `-`           |
| `user_splitting_options`     | `dict` | Options for splitting data by user.                                                     | Refer Below    | `-`           |
| `stream_aggregation_options` | `dict` | Options for aggregating data by stream.                                                 | Refer Below    | `-`           |
| `preprocessing_options`      | `dict` | Options for preprocessing the data.                                                     | `-`           | `-`           |
| `dfencoder_options`          | `dict` | Options for configuring the data frame encoder, used for training the model.            | Refer Below    | `-`           |
| `mlflow_writer_options`      | `dict` | Options for the MLflow model writer, which is responsible for saving the trained model. | Refer Below    | `-`           |

### `batching_options`

| Key                      | Type  | Description                              | Example Value                                           | Default Value |
|--------------------------|-------|------------------------------------------|---------------------------------------------------------|---------------|
| `end_time`               | `str`   | End time of the time range to process.   | `"2023-03-01T00:00:00"`                                   | `-`           |
| `iso_date_regex_pattern` | `str`   | ISO date regex pattern.                  | `"\\\\d{4}-\\\\d{2}-\\\\d{2}T\\\\d{2}:\\\\d{2}:\\\\d{2}"` | `-`           |
| `parser_kwargs`          | `dict`  | Keyword arguments to pass to the parser. | `{}`                                                      | `-`           |
| `period`                 | `str`   | Time period to batch the data.           | `"1min"`                                                  | `-`           |
| `sampling_rate_s`        | `float` | Sampling rate in seconds.                | `60`                                                      | `-`           |
| `start_time`             | `str`   | Start time of the time range to process. | `"2023-02-01T00:00:00"`                                   | `-`           |

### `user_splitting_options`

| Key                  | Type      | Description                                           | Example Value | Default Value |
|----------------------|-----------|-------------------------------------------------------|---------------|---------------|
| `fallback_username`  | `str`       | Fallback user to use if no model is found for a user. | `"generic"`     | `-`           |
| `include_generic`    | `bool`      | Include generic models in the results.                | `true`          | `-`           |
| `include_individual` | `bool`      | Include individual models in the results.             | `true`          | `-`           |
| `only_users`         | `list[str]` | List of users to include in the results.              | `[]`            | `-`           |
| `skip_users`         | `list[str]` | List of users to exclude from the results.            | `[]`            | `-`           |
| `userid_column_name` | `str`       | Column name for the user ID.                          | `"user_id"`     | `-`           |


### `stream_aggregation_options`

| Key                     | Type   | Description                                                 | Example Value | Default Value |
|-------------------------|--------|-------------------------------------------------------------|---------------|---------------|
| `cache_mode`            | `str` | Mode for managing user cache. Setting to `batch` flushes cache once trigger conditions are met. Otherwise, continue to aggregate user's history. | `"batch"`       | `"batch"`       |
| `min_history`           | `int`    | Minimum history to trigger a new training event             | `1`             | `1`           |
| `max_history`           | `int`    | Maximum history to include in a new training event          | `0`             | `0`           |
| `timestamp_column_name` | `str` | Name of the column containing timestamps                    | `'timestamp'`   | `'timestamp'`   |
| `aggregation_span`      | `str` | Look back time span for training data in a new training event | `"60d"`         | `60d`         |
| `cache_to_disk`         | `bool`   | Whether or not to cache streaming data to disk              | `false`         | `false`       |
| `cache_dir`             | `str` | Directory to use for caching streaming data                 | `"./.cache"`    | `"./.cache"`    |

### `dfencoder_options`

| Parameter         | Type  | Description                            | Example Value                                                                                                                                                                                                                                                 | Default Value |
|-------------------|-------|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `feature_columns` | `list`  | List of feature columns to train on    | `["column1", "column2", "column3"]`| `- `          |
| `epochs`          | `int`   | Number of epochs to train for          | `50` | `-`           |
| `model_kwargs`    | `dict`  | Keyword arguments to pass to the model | `{"encoder_layers": [64, 32], "decoder_layers": [32, 64], "activation": "relu", "swap_p": 0.1, "lr": 0.001, "lr_decay": 0.9, "batch_size": 32, "verbose": 1, "optimizer": "adam", "scalar": "min_max", "min_cats": 10, "progress_bar": false, "device": "cpu"}` | `-`             |
| `validation_size` | `float` | Size of the validation set             | `0.1` | `-`           |

### `mlflow_writer_options`

| Key                         | Type       | Description                       | Example Value                 | Default Value |
|-----------------------------|------------|-----------------------------------|-------------------------------|---------------|
| `conda_env`                 | `str`     | Conda environment for the model   | `"path/to/conda_env.yml"`       | `[Required]`  |
| `databricks_permissions`    | `dict` | Permissions for the model         | -                    | `None`        |
| `experiment_name_formatter` | `str`     | Formatter for the experiment name | `"experiment_name_{timestamp}"` | `[Required]`  |
| `model_name_formatter`      | `str`     | Formatter for the model name      | `"model_name_{timestamp}"`      | `[Required]`  |
| `timestamp_column_name`     | `str`     | Name of the timestamp column      | `"timestamp"`                   | `"timestamp"`   |
