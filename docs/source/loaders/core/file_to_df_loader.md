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

## File to DataFrame Loader

[DataLoader](../../modules/core/data_loader.md) module is used to load data files content into a DataFrame using custom loader function. This loader function can be configured to use different processing methods, such as `"single_thread"`, `"dask"`, or `"dask_thread"`, as determined by the `MORPHEUS_FILE_DOWNLOAD_TYPE` environment variable. When `download_method` is `"dask"`, or `"dask_thread"`, a Dask client is created to process the files, otherwise, a single thread is used.

After processing, the resulting DataFrame is cached using a hash of the file paths. This loader also has the ability to load file content from S3 buckets, in addition to loading data from the disk.

### Example Loader Configuration

Using below configuration while loading DataLoader module, specifies that the DataLoader module should utilize the `file_to_df` loader when loading files into a DataFrame.

```json
{
	"loaders": [{
		"id": "file_to_df"
	}]
}
```

**Note** :  Loaders can receive configuration from the `load` task via [control message](../../developer_guide/guides/9_control_messages.md) during runtime.

### Task Configurable Parameters

The parameters that can be configured for this specific loader at load task level:

| Parameter          | Type       | Description                      | Example Value            | Default Value  |
| ------------------ | ---------- | -------------------------------- | ------------------------ | -------------- |
| `batcher_config  ` | dictionary | Options for batching             | Refer Below                | `[Required]`   |
| `files`            | array      | List of files to load            | `["/path/to/input/files"]` | `[]`           |
| `loader_id`        | string     | Unique identifier for the loader | `"file_to_df"`             | `[Required]`   |


### `batcher_config`

| Key                     | Type       | Description                                | Example Value        | Default Value |
|-------------------------|------------|--------------------------------------------|----------------------|---------------|
| `cache_dir`             | string     | Directory to cache the rolling window data | `"/path/to/cache"`     | `-`           |
| `file_type`             | string     | Type of the input file                     | `"csv"`                | `"JSON"`      |
| `filter_null`           | boolean    | Whether to filter out null values          | `true`                 | `false`       |
| `parser_kwargs`         | dictionary | Keyword arguments to pass to the parser    | `{"delimiter": ","}`   | `-`           |
| `schema`                | dictionary | Schema of the input data                   | Refer Below            | `-`           |
| `timestamp_column_name` | string     | Name of the timestamp column               | `"timestamp"`          | `-`           |

### Example Load Task Configuration

Below JSON configuration specifies how to pass additional configuration to the loader through a control message task at runtime.

```json
{
	"type": "load",
	"properties": {
		"loader_id": "file_to_df",
		"files": ["/path/to/input/files"],
		"batcher_config": {
			"timestamp_column_name": "timestamp_column_name",
			"schema": "string",
			"file_type": "JSON",
			"filter_null": false,
			"parser_kwargs": {
				"delimiter": ","
			},
			"cache_dir": "/path/to/cache"
		}
	}
}
```

**Note** : The {py:mod}`~morpheus.modules.file_batcher` module currently generates tasks internally and assigns them to control messages, and then sends them to [DataLoader](../../modules/core/data_loader.md) module which uses {py:mod}`~morpheus.loaders.file_to_df_loader`. Having stated that, this loader configuration is obtained from the [File Batcher](../../modules/core/file_batcher.md) module configuration.
