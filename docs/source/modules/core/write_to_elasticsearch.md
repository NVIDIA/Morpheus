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

## Write to Elasticsearch Module

This module reads an input data stream, converts each row of data to a document format suitable for Elasticsearch, and writes the documents to the specified Elasticsearch index using the Elasticsearch bulk API.

### Configurable Parameters

| Parameter               | Type         | Description                                                                                             | Example Value                 | Default Value |
|-------------------------|--------------|---------------------------------------------------------------------------------------------------------|-------------------------------|---------------|
| `index`                 | `str`          | Elasticsearch index.                                                                                   | `"my_index"`                    | `[Required]`             |
| `connection_kwargs`     | `dict`         | Elasticsearch connection keyword arguments configuration.                                                        | `{"hosts": ["host": "localhost", ...}`    | `[Required]`             |
| `raise_on_exception`    | `bool`         | Raise or suppress exceptions when writing to Elasticsearch.                                           | `true`                          | `false`         |
| `pickled_func_config`   | `str`          | Pickled custom function configuration to update `connection_kwargs` as needed for the client connection. | Refer Below     | `None`          |
| `refresh_period_secs`   | `int`          | Time in seconds to refresh the client connection.                                                      | `3600`                          | `2400`          |

### Example JSON Configuration

```json
{
  "index": "test_index",
  "connection_kwargs": {"hosts": [{"host": "localhost", "port": 9200, "scheme": "http"}]},
  "raise_on_exception": true,
  "pickled_func_config": {
    "pickled_func_str": "pickled function as a string",
    "encoding": "latin1"
  },
  "refresh_period_secs": 2400
}
```
