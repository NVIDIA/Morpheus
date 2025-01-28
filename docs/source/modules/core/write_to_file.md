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

## WriteToFile Module

This module writes messages to a file.

### Configurable Parameters

| Parameter           | Type      | Description                              | Example Value | Default Value    |
|---------------------|-----------|------------------------------------------|---------------|------------------|
| `filename`          | string    | Path to the output file                  | `"output.csv"`  | `None`           |
| `file_type`         | string    | Type of file to write                    | `"CSV"`         | `AUTO`           |
| `flush`             | boolean      | If true, flush the file after each write | `false`         | `false `         |
| `include_index_col` | boolean      | If true, include the index column        | `false`         | `true`           |
| `overwrite`         | boolean      | If true, overwrite the file if it exists | `true`          | `false`          |

### Example JSON Configuration

```json
{
  "filename": "output.csv",
  "overwrite": true,
  "flush": false,
  "file_type": "CSV",
  "include_index_col": false
}
```
