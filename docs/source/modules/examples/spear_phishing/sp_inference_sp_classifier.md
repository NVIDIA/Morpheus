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

## Spear Phishing Inference Module

Module ID: `inference`
Module Namespace: `morpheus_spear_phishing`

This module defines a setup for spear-phishing inference.

### Configurable Parameters

| Parameter              | Type | Description                           | Example Value      | Default Value |
|------------------------|------|---------------------------------------|--------------------|---------------|
| `tracking_uri`         | string  | The tracking URI for the model        | `"/path/to/uri"`     | `None`        |
| `registered_model`     | string  | The registered model for inference    | `"model_1"`          | `None`        |
| `input_model_features` | list | The input features for the model      | `["feat1", "feat2"]` | `[]`          |
| `raise_on_failure`     | boolean | If true, raise exceptions on failures | `false`              | `false`       |

### Example JSON Configuration

```json
{
  "tracking_uri": "/path/to/uri",
  "registered_model": "model_1",
  "input_model_features": ["feat1", "feat2"],
  "raise_on_failure": false
}
