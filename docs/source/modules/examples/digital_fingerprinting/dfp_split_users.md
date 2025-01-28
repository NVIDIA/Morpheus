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

## DFP Split Users Module

This module function splits the data based on user IDs.

### Configurable Parameters

| Key                   | Type | Description                                          | Example Value               | Default Value  |
|-----------------------|------|------------------------------------------------------|-----------------------------|----------------|
| `fallback_username`     | `str`  | The user ID to use if the user ID is not found       | `"generic_user"`              | `"generic_user"` |
| `include_generic`       | `bool` | Whether to include a generic user ID in the output   | `false`                       | `false`        |
| `include_individual`    | `bool` | Whether to include individual user IDs in the output | `true`                        | `false`        |
| `only_users`            | `list` | List of user IDs to include; others will be excluded | `["user1", "user2", "user3"]` | `[]`           |
| `skip_users`            | `list` | List of user IDs to exclude from the output          | `["user4", "user5"]`          | `[]`           |
| `timestamp_column_name` | `str`  | Name of the column containing timestamps             | `"timestamp"`                 | `"timestamp"`    |
| `userid_column_name`    | `str`  | Name of the column containing user IDs               | `"username"`                  | `"username"`     |

### Example JSON Configuration

```json
{
  "fallback_username": "generic_user",
  "include_generic": false,
  "include_individual": true,
  "only_users": [
    "user1",
    "user2",
    "user3"
  ],
  "skip_users": [
    "user4",
    "user5"
  ],
  "timestamp_column_name": "timestamp",
  "userid_column_name": "username"
}
```
