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

## Spear Phishing Email Enrichment Module

Module ID: `email_enrichment`
Module Namespace: `morpheus_spear_phishing`

This module performs spear phishing email enrichment.

### Configurable Parameters

| Parameter                | Type | Description                                                         | Example Value          | Default Value |
|--------------------------|------|---------------------------------------------------------------------|------------------------|---------------|
| `sender_sketches`        | list | List of sender strings naming sender sketch inputs.                 | `["sender1", "sender2"]` | `[]`          |
| `intents`                | list | List of intent strings naming computed intent inputs.               | `["intent1", "intent2"]` | `[]`          |
| `raise_on_failure`       | boolean | Indicate if we should treat processing errors as pipeline failures. | `false`                  | `false`       |
| `token_length_threshold` | integer  | Minimum token length to use when computing syntax similarity        | `5`                      | `None`          |

### Example JSON Configuration

```json
{
  "sender_sketches": ["sender1", "sender2"],
  "intents": ["intent1", "intent2"],
  "raise_on_failure": false,
  "token_length_threshold": 5
}
