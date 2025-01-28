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

## File to DataFrame Module

This module reads data from the batched files into a DataFrame after receiving input from the `FileBatcher` module. In
addition to loading data from the disk, it has the ability to load the file content from S3 buckets.

### Configurable Parameters

| Parameter               | Type       | Description                                | Example Value        | Default Value |
|-------------------------|------------|--------------------------------------------|----------------------|---------------|
| `cache_dir`             | string     | Directory to cache the rolling window data | `"/path/to/cache"`     | `-`           |
| `file_type`             | string     | Type of the input file                     | `"csv"`                | `"JSON"`      |
| `filter_null`           | boolean    | Whether to filter out null values          | `True`                 | `False`       |
| `parser_kwargs`         | dictionary | Keyword arguments to pass to the parser    | `{"delimiter": ","}`   | `-`           |
| `schema`                | dictionary | Schema of the input data                   | Refer Below            | `-`           |
| `timestamp_column_name` | string     | Name of the timestamp column               | `"timestamp"`          | `-`           |

### Example JSON Configuration

```json
{
  "cache_dir": "/path/to/cache",
  "file_type": "csv",
  "filter_null": true,
  "parser_kwargs": {
    "delimiter": ","
  },
  "schema": {
    "schema_str": "string",
    "encoding": "latin1"
  },
  "timestamp_column_name": "timestamp"
}
```
