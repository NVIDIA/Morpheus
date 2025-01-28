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

## Serialize Module

This module filters columns from a `ControlMessage` object, emitting a `MessageMeta`.

### Configurable Parameters

| Parameter       | Type         | Description                                                  | Example Value                       | Default Value         |
|-----------------|--------------|--------------------------------------------------------------|-------------------------------------|-----------------------|
| `columns`       | list[string] | List of columns to include                                   | `["column1", "column2", "column3"]`   | `None`                |
| `exclude`       | list[string] | List of regex patterns to exclude columns                    | `["column_to_exclude"]`               | `[r'^ID$', r'^_ts_']` |
| `fixed_columns` | boolean         | If true, the columns are fixed and not determined at runtime | `true`                                | `true`                |
| `include`       | string       | Regex to include columns                                     | `"^column"`                           | `None`                |
| `use_cpp`       | boolean         | If true, use C++ to serialize                                | `true`                                | `false`               |

### Example JSON Configuration

```json
{
  "include": "^column",
  "exclude": ["column_to_exclude"],
  "fixed_columns": true,
  "columns": ["column1", "column2", "column3"],
  "use_cpp": true
}
```
