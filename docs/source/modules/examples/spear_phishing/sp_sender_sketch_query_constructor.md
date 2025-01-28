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


## SQL Sender Sketch Query Constructor Module

Module ID: `sql_sender_sketch_query_constructor`
Module Namespace: `morpheus_spear_phishing`

This module constructs SQL sender sketch queries in the spear phishing detection pipeline.

### Configurable Parameters

| Parameter          | Type | Description                           | Example Value | Default Value |
|--------------------|------|---------------------------------------|---------------|---------------|
| `raise_on_failure` | boolean | If true, raise exceptions on failures | `false`         | `false`       |

### Example JSON Configuration

```json
{
  "raise_on_failure": false
}
```
