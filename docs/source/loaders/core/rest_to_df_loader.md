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

## REST to DataFrame Loader

[DataLoader](../../modules/core/data_loader.md) module is used to load data files content into a DataFrame using custom loader function. This loader function can be configured to send REST requests with customized parameters to retrieve data from endpoints. Refer Below for the specific configuration format.

### Example Loader Configuration

Using below configuration while loading DataLoader module, specifies that the DataLoader module should utilize the `rest` loader when loading files into a DataFrame.

```json
{
	"loaders": [{
		"id": "rest"
	}]
}
```

**Note** :  Loaders can receive configuration from the `load` task via [control message] during runtime.

### Task Configurable Parameters

The parameters that can be configured for this specific loader at load task level:

| Parameter   | Type   | Description                         | Example Value | Default Value |
| ----------- | ------ | ----------------------------------- | ------------- | ------------- |
| `loader_id` | string | Unique identifier for the loader    | `"rest"`        | `[Required]`  |
| `strategy`  | string | Strategy for constructing DataFrame | `"aggregate"`   | `[Required]`  |
| `queries`   | array  | parameters of REST queries          | Refer Below     | `[Required]`  |


### `queries`

| Key            | Type       | Description                                                       | Example Value                                                                  | Default Value |
| -------------- | ---------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------- |
| `method`       | string     | Method of request                                                 | `"GET"`                                                                          | `"GET"`       |
| `endpoint`     | string     | Endpoint of request                                               | `"0.0.0.0/path/to/target?param1=true"`                                           | `[Required]`  |
| `port`         | string     | Target port of request                                            | `"80"`                                                                           | `"80"`        |
| `http_version` | string     | HTTP version of request                                           | `"1.1"`                                                                          | `"1.1"`       |
| `content_type` | string     | Content type of request body in a POST request                    | `"text/plain"`                                                                   | `-`           |
| `body`         | string     | Request body in a POST request                                    | `"param1=true&param2=false"`                                                     | `-`           |
| `X-Headers`    | dictionary | Customized X-Headers of request                                   | `'{"X-Header1":"header1", "X-Header2":"header2"}'`                               | `-`           |
| `params`       | array      | Parameters of requested URL, override values included in endpoint | `'[{"param1": "true", "param2":"false"}, {"param1": "false", "param2":"true"}]'` | `-`           |


### Example Load Task Configuration

Below JSON configuration specifies how to pass additional configuration to the loader through a control message task at runtime.

```json
{
    "type":"load",
    "properties":
    {
        "loader_id":"rest",
        "strategy":"aggregate",
        "queries":[
            {
                "method":"<GET/POST>",
                "endpoint":"0.0.0.0/?param1=false&param2=true",
                "port": "80",
                "http_version": "1.1",
                "content_type":"text/plain",
                "body":"http POST body",
                "x-headers":
                {
                    "X-Header1":"header1",
                    "X-Header2":"header2"
                },
                "params":
                [
                    {
                        "param1":"true"
                    },
                    {
                        "param1":"false",
                        "param2":"true"
                    }
                ]
            }
        ]
    }
}
```
