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

## `dfp_preproc`

This module function allows for the consolidation of multiple DFP pipeline modules relevant to inference/training
process into a single module.

### Configurable Parameters

| Parameter                | Type       | Description                                      | Example Value | Default Value  |
|--------------------------|------------|--------------------------------------------------|---------------|----------------|
| `cache_dir`              | string     | Directory used for caching intermediate results. | `"/tmp/cache"`  | `-`            |
| `timestamp_column_name`  | string     | Name of the column containing timestamps.        | `"timestamp"`   | `-`            |
| `pre_filter_options`     | dictionary | Options for pre-filtering control messages.      | Refer Below    | `-`            |
| `batching_options`       | dictionary | Options for batching files.                      | Refer Below    | `-`            |
| `user_splitting_options` | dictionary | Options for splitting data by user.              | Refer Below    | `-`            |
| `supported_loaders`      | dictionary | Supported data loaders for different file types. | -             | `-`            |

#### `pre_filter_options`

| Parameter               | Type    | Description                           | Example Value | Default Value |
|-------------------------|---------|---------------------------------------|---------------|---------------|
| `enable_task_filtering` | boolean | Enables filtering based on task type. | `true`          | `-`           |
| `filter_task_type`      | string  | The task type to be used as a filter. | `"task_a"`      | `-`           |
| `enable_data_filtering` | boolean | Enables filtering based on data type. | `true`          | `-`           |
| `filter_data_type`      | string  | The data type to be used as a filter. | `"type_a"`      | `-`           |

#### `batching_options`

| Parameter                | Type       | Description                              | Example Value                          | Default Value |
|--------------------------|------------|------------------------------------------|----------------------------------------|---------------|
| `end_time`               | string     | End time of the time range to process.   | `"2022-01-01T00:00:00Z"`                 | `-`           |
| `iso_date_regex_pattern` | string     | ISO date regex pattern.                  | `"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"` | `-`           |
| `parser_kwargs`          | dictionary | Keyword arguments to pass to the parser. | `{}`                                     | `-`           |
| `period`                 | string     | Time period to batch the data.           | `"1D"`                                   | `-`           |
| `sampling_rate_s`        | float      | Sampling rate in seconds.                | `"1.0"`                                  | `-`           |
| `start_time`             | string     | Start time of the time range to process. | `"2021-01-01T00:00:00Z"`                 | `-`           |

#### `user_splitting_options`

| Parameter            | Type    | Description                                           | Example Value          | Default Value |
|----------------------|---------|-------------------------------------------------------|------------------------|---------------|
| `fallback_username`  | string  | Fallback user to use if no model is found for a user. | `"generic"`              | `-`           |
| `include_generic`    | boolean | Include generic models in the results.                | `true`                 | `-`           |
| `include_individual` | boolean | Include individual models in the results.             | `true`                 | `-`           |
| `only_users`         | list    | List of users to include in the results.              | `["user_a", "user_b"]`   | `-`           |
| `skip_users`         | list    | List of users to exclude from the results.            | `["user_c"]`             | `-`           |
| `userid_column_name` | string  | Column name for the user ID.                          | `"user_id"`              | `-`           |

### Example JSON Configuration

```json
{
  "cache_dir": "/tmp/cache",
  "timestamp_column_name": "timestamp",
  "pre_filter_options": {
    "enable_task_filtering": true,
    "filter_task_type": "task_a",
    "enable_data_filtering": true,
    "filter_data_type": "type_a"
  },
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
  "supported_loaders": {}
}
```
