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

## Data Loader Module

This module takes a control message and attempts to process any `load` tasks in the message. The module itself is
configured to use a set of loaders, each of which is responsible for loading a specific type of data. These loaders
are specified in the module configuration file at the time of object construction.

### Configurable Parameters

| Parameter | Type  | Description                                       | Example Value |  Default Value  |
|-----------|-------|---------------------------------------------------|---------------|-----------------|
| `loaders` | array | An array containing information on loaders to use | Refer Below     | `[]`            |

### `loaders`

| Parameter    | Type       | Description                              | Example Value                          | Default Value |
|--------------|------------|------------------------------------------|----------------------------------------|---------------|
| `id`         | string     | Unique identifier for the loader         | "loader1"                              | `-`           |
| `properties` | dictionary | Dictionary of properties for that loader | {... loader specific parameters ...}   | `{}`          |

### Example JSON Configuration

```
{
  "loaders": [
    {
      "id": "loader1",
      "properties": {
        ... loader specific parameters ...
      }
    },
    {
      "id": "loader2",
      "properties": {
        ... loader specific parameters ...
      }
    }
  ]
}
```
