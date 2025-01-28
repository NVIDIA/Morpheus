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

## Batch Data Payload Module

This module batches incoming control message data payload into smaller batches based on the specified configurations.

### Configurable Parameters

| Parameter                   | Type       | Description                       | Example Value                   | Default Value |
|-----------------------------|------------|-----------------------------------|---------------------------------|---------------|
| `max_batch_size`            | integer        | The maximum size of each batch    | 256                             | `256`        |
| `raise_on_failure`          | boolean       | Whether to raise an exception if a failure occurs during processing | false | `false` |
| `group_by_columns`          | list       | The column names to group by when batching | `["col1", "col2"]`                     | `[]`            |
| `disable_max_batch_size`    | boolean       | Whether to disable the `max_batch_size` and only batch by group     | false         | `false`         |
| `timestamp_column_name`     | string        | The name of the timestamp column  | None                          | `None`          |
| `timestamp_pattern`         | string        | The pattern to parse the timestamp column | None                    | `None`          |
| `period`                    | string        | The period for grouping by timestamp | H                          | `D`           |


### Example JSON Configuration

```json
{
  "max_batch_size": 256,
  "raise_on_failure": false,
  "group_by_columns": [],
  "disable_max_batch_size": false,
  "timestamp_column_name": null,
  "timestamp_pattern": null,
  "period": "D"
}
```
