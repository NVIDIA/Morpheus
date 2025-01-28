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

## Inference Intent

Module ID: `infer_email_intent`
Module Namespace: `morpheus_spear_phishing`

Infers an 'intent' for a given email body.

### Configurable Parameters

| Parameter          | Type | Description                             | Example Value         | Default Value           |
|--------------------|------|-----------------------------------------|-----------------------|-------------------------|
| `intent`           | string  | The intent for the model                | `"classify"`            | `None`                  |
| `task`             | string  | The task for the model                  | `"text-classification"` | `"text-classification"` |
| `model_path`       | string  | The path to the model                   | `"/path/to/model"`      | `None`                  |
| `truncation`       | boolean | If true, truncates inputs to `max_length` | `true`                  | `true`                  |
| `max_length`       | integer  | Maximum length for model input          | `512`                   | `512`                   |
| `batch_size`       | integer  | The size of batches for processing      | `256`                   | `256`                   |
| `feature_col`      | string  | The feature column to use               | `"body"`                | `"body"`                |
| `label_col`        | string  | The label column to use                 | `"label"`               | `"label"`               |
| `device`           | integer  | The device to run on                    | `0`                     | `0`                     |
| `raise_on_failure` | boolean | If true, raise exceptions on failures   | `false`                 | `false`                 |

### Example JSON Configuration

```json
{
  "intent": "classify",
  "task": "text-classification",
  "model_path": "/path/to/model",
  "truncation": true,
  "max_length": 512,
  "batch_size": 256,
  "feature_col": "body",
  "label_col": "label",
  "device": 0,
  "raise_on_failure": false
}
