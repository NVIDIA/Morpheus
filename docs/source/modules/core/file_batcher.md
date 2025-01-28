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

## File Batcher Module

This module loads the input files, removes files that are older than the chosen window of time, and then groups the
remaining files by period that fall inside the window.

### Configurable Parameters

| Parameter               | Type       | Description                   | Example Value          | Default Value |
|-------------------------|------------|-------------------------------|------------------------|---------------|
| `batching_options`      | dictionary | Options for batching          | Refer Below              | `-`           |
| `cache_dir`             | string     | Cache directory               | "./file_batcher_cache" | `None`        |
| `file_type`             | string     | File type                     | "JSON"                 | `"JSON"`      |
| `filter_nulls`          | boolean    | Whether to filter null values | false                  | `false`       |
| `schema`                | dictionary | Data schema                   | Refer Below              | `[Required]`  |
| `timestamp_column_name` | string     | Name of the timestamp column  | "timestamp"            | `"timestamp"` |

### `batching_options`

| Key                      | Type            | Description                         | Example Value                               | Default Value              |
|--------------------------|-----------------|-------------------------------------|---------------------------------------------|----------------------------|
| `end_time`               | datetime/string | End of the time window          | "2023-03-14T23:59:59"                       | `None`                     |
| `iso_date_regex_pattern` | string          | Regex pattern for ISO date matching | "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}" | `<iso_date_regex_pattern>` |
| `parser_kwargs`          | dictionary      | Additional arguments for the parser | {}                                          | `{}`                       |
| `period`                 | string          | Time period for grouping files      | "1d"                                        | `"D"`                      |
| `sampling_rate_s`        | integer         | Sampling rate in seconds            | 0                                          | `None`                       |
| `start_time`             | datetime/string | Start time of the time window       | "2023-03-01T00:00:00"                       | `None`                     |

### `schema`

| Key          | Type   | Description   | Example Value | Default Value |
|--------------|--------|---------------|---------------|---------------|
| `encoding`   | string | Encoding      | "latin1"      | `"latin1"`    |
| `schema_str` | string | Schema string | "string"      | `[Required]`  |

### Example JSON Configuration

```json
{
  "batching_options": {
    "end_time": "2023-03-14T23:59:59",
    "iso_date_regex_pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}",
    "parser_kwargs": {},
    "period": "1d",
    "sampling_rate_s": 60,
    "start_time": "2023-03-01T00:00:00"
  },
  "cache_dir": "./file_batcher_cache",
  "file_type": "JSON",
  "filter_nulls": false,
  "schema": {
    "schema_str": "string",
    "encoding": "latin1"
  },
  "timestamp_column_name": "timestamp"
}
```
