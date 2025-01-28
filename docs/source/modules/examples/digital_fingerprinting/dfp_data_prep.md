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

## DFP Data Prep Module

This module function prepares data for either inference or model training.

### Configurable Parameters

| Parameter               | Type   | Description                  | Example Value | Default Value |
|-------------------------|--------|------------------------------|---------------|---------------|
| `schema`                | dict   | Schema configuration         | Refer Below    | `-`           |
| `timestamp_column_name` | string | Name of the timestamp column | `"timestamp"`   | `timestamp`   |

#### `schema`

| Key                  | Type   | Description                      | Example Value           | Default Value |
|----------------------|--------|----------------------------------|-------------------------|---------------|
| `schema_str`         | string | Serialized schema string         | `"cPickle schema string"` | `-`           |
| `encoding`           | string | Encoding used for the `schema_str` | `"latin1"`                | `-`           |
| `input_message_type` | string | Pickled message type             | `"message type"`          | `-`           |

### Example JSON Configuration

```json
{
  "schema": {
    "schema_str": "cPickle schema string",
    "encoding": "latin1",
    "input_message_type": "message type"
  },
  "timestamp_column_name": "timestamp"
}
```
