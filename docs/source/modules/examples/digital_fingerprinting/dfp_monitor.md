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

## DFP Monitor Module

This module function monitors the pipeline message flow rate.

### Configurable Parameters

| Key                         | Type    | Description                                                | Example Value | Default Value |
| ----------------------------|---------|------------------------------------------------------------|---------------|---------------|
| `description`               | string  | Name to show for this Monitor Stage in the console window  | `"Progress"`    | `"Progress"`    |
| `silence_monitors`          | boolean    | Silence the monitors on the console                        | `True`     | `False`        |
| `smoothing`                 | float   | Smoothing parameter to determine how much the throughput should be averaged | `0.01` | `0.05` |
| `unit`                      | string  | Units to show in the rate value                             | `"messages"`    | `"messages"`    |
| `delayed_start`             | boolean    | When enabled, the progress bar will not be shown until the first message is received. Otherwise, the progress bar is shown on pipeline startup and will begin timing immediately. In large pipelines, this option may be desired to give a more accurate timing. | `True`  | `False`   |
| `determine_count_fn_schema` | string  | Custom function for determining the count in a message      | `"Progress"`    | `"Progress"`    |
| `log_level`                 | string  | Enable this stage when the configured log level is at `log_level` or lower. | `"DEBUG"` | `"INFO"` |
