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

## Filesystem Spec Loader

[DataLoader](../../modules/core/data_loader.md) module is configured to use this loader function. It is responsible for loading data from external sources using the [`fsspec`](https://filesystem-spec.readthedocs.io/en/latest/) library, and returns the updated ControlMessage object with payload as MessageMeta, which contains DataFrame (with filenames).


### Example Loader Configuration

```json
{
	"loaders": [{
		"id": "fsspec"
	}]
}
```

**Note** :  Loaders can receive configuration from the `load` task via [control message](../../developer_guide/guides/9_control_messages.md) during runtime.

### Task Configurable Parameters

The parameters that can be configured for this specific loader at load task level:

| Parameter          | Type       | Description                      | Example Value                     | Default Value  |
| ------------------ | ---------- | -------------------------------- | --------------------------------- | -------------- |
| `files`            | array      | List of files to load            | `["/your/input/filepath"]`      	 | `[]`           |
| `loader_id`        | string     | Unique identifier for the loader | `"file_to_df"`                      | `[Required]`   |




### Example Load Task Configuration

Below JSON configuration specifies how to pass additional configuration to the loader through a control message task at runtime.

```json
{
	"type": "load",
	"properties": {
		"loader_id": "file_to_df",
		"files": ["/your/input/filepath"],
	}
}
```
