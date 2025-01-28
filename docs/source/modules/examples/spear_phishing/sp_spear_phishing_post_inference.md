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

## Pre-inference Module

Module ID: `post_inference`
Module Namespace: `morpheus_spear_phishing`

This module represents the post-inference phase of the spear phishing inference pipeline. It handles the output from the
label and score module, updates the sender sketch, and prepares the final output.

## Configurable Parameters

| Parameter              | Type | Description                                                                                                  |
|------------------------|------|--------------------------------------------------------------------------------------------------------------|
| `scoring_config`       | dictionary | Configuration for scoring, can include custom parameters for the scoring module. Refer below for more details. |
| `sender_sketch_config` | dictionary | Configuration for sender sketch module, including parameters such as endpoint details and sketch settings.   |

#### `scoring_config`

| Key                | Type  | Description                                                        |
|--------------------|-------|--------------------------------------------------------------------|
| `threshold`        | float | Detection threshold for scoring.                                   |
| `scoring_type`     | string   | Type of scoring to use. Currently only `"probability"` is supported. |
| `raise_on_failure` | boolean  | If true, raise exceptions on failures. Default is False.           |

#### `sender_sketch_config`

| Key                           | Type | Description                                                  | Default Value |
|-------------------------------|------|--------------------------------------------------------------|---------------|
| `endpoint`                    | dictionary | See `endpoint` sub-parameters                                 | `None`        |
| `sender_sketches`             | list | List of sender sketches                                      | `[]`          |
| `required_intents`            | list | List of required intents                                     | `[]`          |
| `raise_on_failure`            | boolean | If true, raise exceptions on failures                        | `False`       |
| `token_length_threshold`      | int  | Minimum token length to use when computing syntax similarity | `3`           |
| `sender_sketch_tables_config` | dictionary | Configuration for sender sketch tables                       | `None`        |

##### `endpoint`

| Key          | Type | Description                                |
|--------------|------|--------------------------------------------|
| `database`   | string  | Sender sketch database name                |
| `drivername` | string  | Driver name for the sender sketch database |
| `host`       | string  | Host of the sender sketch database         |
| `port`       | string  | Port of the sender sketch database         |
| `username`   | string  | Username for the sender sketch database    |
| `password`   | string  | Password for the sender sketch database    |
