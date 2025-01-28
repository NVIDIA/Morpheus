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

## SQL Loader

[DataLoader](./../../modules/core/data_loader.md) module is configured to use this loader function. SQL loader to
fetch data from a SQL database and store it in a DataFrame, and returns the updated ControlMessage object with payload
as MessageMeta.

### Example Loader Configuration

```json
{
  "loaders": [
    {
      "id": "SQLLoader"
    }
  ]
}
```

**Note** :  Loaders can receive configuration from the `load` task via ControlMessage during runtime.

### Task Configurable Parameters

The parameters that can be configured for this specific loader at load task level:

| Parameter    | Type       | Description                              | Example Value      | Default Value |
|--------------|------------|------------------------------------------|--------------------|---------------|
| `strategy`   | string     | Strategy for combining queries           | `"aggregate"`      	 | `aggregate`   |
| `loader_id`  | string     | Unique identifier for the loader         | `"file_to_df"`       | `[Required]`  |
| `sql_config` | dictionary | Dictionary containing SQL queries to run | `"file_to_df"`       | Refer Below   |

`sql_config`

| Parameter | Type | Description                                       | Example Value                              | Default Value |
|-----------|------|---------------------------------------------------|--------------------------------------------|---------------|
| `queries` | list | List of dictionaries composing a query definition | `"[query_dict_1, ..., query_dict_n]"`      	 | Refer Below   |

`queries`

| Parameter           | Type       | Description                          | Example Value                                                   | Default Value |
|---------------------|------------|--------------------------------------|-----------------------------------------------------------------|---------------|
| `connection_string` | string     | Strategy for combining queries       | `"postgresql://postgres:postgres@localhost:5432/postgres"`      	 | `[required]`  |
| `query`             | string     | SQL Query to execute                 | `"SELECT * FROM test_table WHERE id IN (?, ?, ?)"`                | `[Required]`  |
| `params`            | dictionary | Named or positional parameters values | `"[foo, bar, baz]"`                                               | `-`           |

### Example Load Task Configuration

Below JSON configuration specifies how to pass additional configuration to the loader through a control message task at
runtime.

```json
{
  "type": "load",
  "properties": {
    "loader_id": "SQLLoader",
    "strategy": "aggregate",
    "sql_config": {
      "queries": [
        {
          "connection_string": "postgresql://postgres:postgres@localhost:5431/postgres",
          "query": "SELECT * FROM test_table WHERE id IN (?, ?, ?)",
          "params": [
            "foo",
            "bar",
            "baz"
          ]
        }
      ]
    }
  }
}
```
