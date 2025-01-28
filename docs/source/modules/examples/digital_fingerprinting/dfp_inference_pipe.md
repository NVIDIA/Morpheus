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

## `dfp_inference_pipe`

This module function allows for the consolidation of multiple DFP pipeline modules relevant to the inference process into a single module.

### Configurable Parameters

| Parameter                    | Type       | Description                                      | Example Value | Default Value |
|------------------------------|------------|--------------------------------------------------|---------------|---------------|
| `batching_options`           | dictionary | Options for batching files.                      | Refer below     | `-`           |
| `cache_dir`                  | string     | Directory used for caching intermediate results. | `"/tmp/cache"`  | `-`           |
| `detection_criteria`         | dictionary | Criteria for filtering detections.               | -             | `-`           |
| `inference_options`          | dictionary | Options for configuring the inference process.   | Refer below     | `-`           |
| `preprocessing_options`      | dictionary | Options for preprocessing data.                  | -             | `-`           |
| `stream_aggregation_options` | dictionary | Options for aggregating data by stream.          | Refer below     | `-`           |
| `timestamp_column_name`      | string     | Name of the column containing timestamps.        | `"timestamp"`   | `-`           |
| `user_splitting_options`     | dictionary | Options for splitting data by user.              | Refer below     | `-`           |
| `write_to_file_options`      | dictionary | Options for writing results to a file.           | -             | `-`           |

#### `batching_options`

| Parameter                | Type   | Description                              | Example Value                                | Default Value |
|--------------------------|--------|------------------------------------------|----------------------------------------------|---------------|
| `end_time`               | string | End time of the time range to process.   | `"2022-01-01T00:00:00Z"`                       | `-`           |
| `iso_date_regex_pattern` | string | ISO date regex pattern.                  | `"\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z"` | `-`           |
| `parser_kwargs`          | dict   | Keyword arguments to pass to the parser. | -                                            | `-`           |
| `period`                 | string | Time period to batch the data.           | `"1D"`                                         | `-`           |
| `sampling_rate_s`        | float  | Sampling rate in seconds.                | `"1.0"`                                        | `-`           |
| `start_time`             | string | Start time of the time range to process. | `"2021-01-01T00:00:00Z"`                       | `-`           |

#### `user_splitting_options`

| Parameter            | Type    | Description                                           | Example Value           | Default Value   |
|----------------------|---------|-------------------------------------------------------|-------------------------|-----------------|
| `fallback_username`  | string  | Fallback user to use if no model is found for a user. | `"generic_user"`          | `"generic_user"`  |
| `include_generic`    | boolean | Include generic models in the results.                | `True`                    | `True`          |
| `include_individual` | boolean | Include individual models in the results.             | `True`                    | `False`         |
| `only_users`         | list    | List of users to include in the results.              | `["user_a","user_b"]`     | `-`             |
| `skip_users`         | list    | List of users to exclude from the results.            | `["user_c"]`              | `-`             |
| `userid_column_name` | string  | Column                                                | `"name for the user ID."` | `"user_id"`       |

### `stream_aggregation_options`

| Parameter               | Type   | Description                                                 | Example Value | Default Value |
|-------------------------|--------|-------------------------------------------------------------|---------------|---------------|
| `cache_mode`            | string | Mode for managing user cache. Setting to `batch` flushes cache once trigger conditions are met. Otherwise, continue to aggregate user's history. | `"batch"`       | `batch`       |
| `min_history`           | int    | Minimum history to trigger a new training event             | `1`             | `1`             |
| `max_history`           | int    | Maximum history to include in a new training event          | `0`             | `0`             |
| `timestamp_column_name` | string | Name of the column containing timestamps                    | `"timestamp"`   | `timestamp`   |
| `aggregation_span`      | string | Look back time span for training data in a new training event | `"60d"`         | `60d`         |
| `cache_to_disk`         | boolean   | Whether or not to cache streaming data to disk              | `False`         | `False`         |
| `cache_dir`             | string | Directory to use for caching streaming data                 | `"./.cache"`    | `"./.cache"`    |

### `inference_options`

| Parameter               | Type   | Description                                          | Example Value           | Default Value   |
|-------------------------|--------|------------------------------------------------------|-------------------------|-----------------|
| `model_name_formatter`  | string | Formatter for model names                            | `"user_{username}_model"` | `[Required]`    |
| `fallback_username`     | string | Fallback user to use if no model is found for a user | `"generic_user"`          | `"generic_user"`  |
| `timestamp_column_name` | string | Name of the timestamp column                         | `"timestamp"`             | `"timestamp"`     |

### Example JSON Configuration

```json
{
  "timestamp_column_name": "timestamp",
  "cache_dir": "/tmp/cache",
  "batching_options": {
    "end_time": "2022-01-01T00:00:00Z",
    "iso_date_regex_pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z",
    "parser_kwargs": {},
    "period": "1D",
    "sampling_rate_s": 1.0,
    "start_time": "2021-01-01T00:00:00Z"
  },
  "user_splitting_options": {
    "fallback_username": "generic",
    "include_generic": true,
    "include_individual": true,
    "only_users": [
      "user_a",
      "user_b"
    ],
    "skip_users": [
      "user_c"
    ],
    "userid_column_name": "user_id"
  },
  "stream_aggregation_options": {
    "timestamp_column_name": "timestamp",
    "cache_mode": "batch",
    "trigger_on_min_history": true,
    "aggregation_span": "1D",
    "trigger_on_min_increment": true,
    "cache_to_disk": false
  },
  "preprocessing_options": {},
  "inference_options": {
    "model_name_formatter": "{model_name}",
    "fallback_username": "generic",
    "timestamp_column_name": "timestamp"
  },
  "detection_criteria": {},
  "write_to_file_options": {}
}
```
