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

## Filter Detections Module

Filter message by a classification threshold.

The Filter Detections module is used to filter rows from a DataFrame based on values in a tensor using a specified
criteria. Rows in the `meta` DataFrame are excluded if their associated value in the `probs` array is less than or equal
to `threshold`.

### Configurable Parameters

| Parameter       | Type       | Description                            | Example Value | Default Value   |
|-----------------|------------|----------------------------------------|---------------|-----------------|
| `copy`          | boolean    | Whether to copy the rows or slice them | `true`          | `true`          |
| `field_name`    | string     | Name of the field to filter on         | `"probs"`       | `"probs"`         |
| `filter_source` | string     | Source of the filter field             | `"AUTO"`        | `"AUTO"`          |
| `schema`        | dictionary | Schema configuration                   | Refer Below     | `-`             |
| `threshold`     | float      | Threshold value to filter on           | `0.5`           | `0.5`           |

### `schema`

| Key                  | Type   | Description          | Example Value         | Default Value |
|----------------------|--------|----------------------|-----------------------|---------------|
| `encoding`           | string | Encoding             | `"latin1"`              | `"latin1"`      |
| `input_message_type` | string | Pickled message type | `"pickle_message_type"` | `[Required]`  |
| `schema_str`         | string | Schema string        | `"string"`              | `[Required]`  |

### Example JSON Configuration

```json
{
  "field_name": "probs",
  "threshold": 0.5,
  "filter_source": "AUTO",
  "copy": true,
  "schema": {
    "input_message_type": "pickle_message_type",
    "encoding": "utf-8"
  }
}
```
