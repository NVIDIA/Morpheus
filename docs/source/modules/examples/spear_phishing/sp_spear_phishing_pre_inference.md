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

Module ID: `pre_inference`
Module Namespace: `morpheus_spear_phishing`

Pre-inference phase of the spear phishing inference pipeline. It loads the necessary modules and establishes the
required connections between modules.

### Configurable Parameters

| Parameter              | Type | Description                              | Default Value |
|------------------------|------|------------------------------------------|---------------|
| `raise_on_failure`     | boolean | If true, raise exceptions on failures    | `False`       |
| `max_batch_size`       | integer  | Maximum size of each batch               | `500`         |
| `intent_config`        | dictionary | See `intent_config` sub-parameters        | `{}`          |
| `sender_sketch_config` | dictionary | See `sender_sketch_config` sub-parameters | `None`        |

#### `intent_config`

| Key                 | Type | Description                     | Default Value |
|---------------------|------|---------------------------------|---------------|
| `required_intents`  | list | List of required intents        | `[]`          |
| `available_intents` | dictionary | Dictionary of available intents | `{}`          |

#### `sender_sketch_config`

| Key                           | Type | Description                                                  | Default Value |
|-------------------------------|------|--------------------------------------------------------------|---------------|
| `endpoint`                    | dictionary | See `endpoint` sub-parameters                                 | `None`        |
| `sender_sketches`             | list | List of sender sketches                                      | `[]`          |
| `required_intents`            | list | List of required intents                                     | `[]`          |
| `raise_on_failure`            | boolean | If true, raise exceptions on failures                        | `False`       |
| `token_length_threshold`      | integer  | Minimum token length to use when computing syntax similarity | `3`           |
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
