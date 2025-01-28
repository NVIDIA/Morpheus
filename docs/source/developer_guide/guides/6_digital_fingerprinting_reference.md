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

# Digital Fingerprinting (DFP) Reference

## Morpheus Configuration
![Morpheus Configuration](img/dfp_deployment_configs.png)

### Pipeline Structure Configuration
![Pipeline Structure Configuration](img/dfp_pipeline_structure.png)

The stages in both the Training and Inference pipelines can be mixed and matched with little impact, that is, the `MultiFileSource` can be configured to pull from S3 or from local files and can be replaced altogether with any other Morpheus input stage. Similarly, the S3 writer can be replaced with any Morpheus output stage. Regardless of the inputs and outputs the core pipeline should remain unchanged. While stages in the core of the pipeline (inside the blue areas in the above diagram) perform common actions that should be configured not exchanged.

### Morpheus `Config`

For both inference and training pipeline the Morpheus `Config` object should be constructed with the same values, for example:
```python
import os

from morpheus.config import Config
from morpheus.config import ConfigAutoEncoder
from morpheus.cli.utils import get_package_relative_file
from morpheus.utils.file_utils import load_labels_file
```
```python
config = Config()
config.num_threads = len(os.sched_getaffinity(0))
config.ae = ConfigAutoEncoder()
config.ae.feature_columns = load_labels_file(get_package_relative_file("data/columns_ae_azure.txt"))
```

Other attributes which might be needed:
| Attribute | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `Config.ae.userid_column_name` | `str` | `userIdentityaccountId` | Column in the `DataFrame` containing the username or user ID |
| `Config.ae.timestamp_column_name` |  `str` | `timestamp` | Column in the `DataFrame` containing the timestamp of the event |
| `Config.ae.fallback_username` | `str` | `generic_user` | Name to use for the generic user model, should not match the name of any real users |

### Schema Definition

#### DataFrame Input Schema (`DataFrameInputSchema`)
The {py:class}`~morpheus.utils.column_info.DataFrameInputSchema` class defines the schema specifying the columns to be included in the output `DataFrame`. Within the DFP pipeline there are two stages where pre-processing is performed, the `DFPFileToDataFrameStage` stage and the `DFPPreprocessingStage`. This decoupling of the pre-processing stages from the actual operations needed to be performed allows for the actual schema to be user-defined in the pipeline and re-usability of the stages. It is up to the user to define the fields which will appear in the `DataFrame`. Any column in the input data that isn't specified in either `column_info` or `preserve_columns` constructor arguments will not appear in the output. The exception to this are JSON fields, specified in the `json_columns` argument which defines JSON fields which are to be normalized.

It is important to note that the fields defined in `json_columns` are normalized prior to the processing of the fields in `column_info`, allowing for processing to be performed on fields nested in JSON columns. For example, say we had a JSON field named `event` containing a key named `timestamp`, which in the JSON data appears as an ISO 8601 formatted date string, we could ensure it was converted to a `datetime` object to downstream stages with the following:
```python
from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.column_info import DateTimeColumn
```
```python
schema = DataFrameInputSchema(
    json_columns=['event'],
    column_info=[DateTimeColumn(name=config.ae.timestamp_column_name, dtype=datetime, input_name='event.timestamp')])
```

In the above examples, three operations were performed:
1. The `event` JSON field was normalized, resulting in new fields prefixed with `event.` to be included in the output `DataFrame`.
2. The newly created field `event.timestamp` is parsed into a `datetime` field.
3. Since the DFP pipeline explicitly requires a timestamp field, we name this new column with the `config.ae.timestamp_column_name` attribute ensuring it matches the pipeline configuration. When `name` and `input_name` are the same the old field is overwritten, and when they differ a new field is created.

The `DFPFileToDataFrameStage` is executed first and is responsible for flattening potentially nested JSON data and performing any sort of data type conversions. The `DFPPreprocessingStage` is executed later after the `DFPSplitUsersStage` allowing for the possibility of per-user computed fields such as the `logcount` and `locincrement` fields mentioned previously. Both stages are performed after the `DFPFileBatcherStage` allowing for per time period (per-day by default) computed fields.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `json_columns` | `List[str]` | Optional list of JSON columns in the incoming `DataFrame` to be normalized (currently using the [`pandas.json_normalize`](https://pandas.pydata.org/docs/reference/api/pandas.json_normalize.html) method). Each key in a JSON field will be flattened into a new column named `<field>.<key>` for example a JSON field named `user` containing a key named `id` will result in a new column named `user.id`. By default, this is an empty `list`. |
| `column_info` | `List[str]` | Optional list of `ColumnInfo` instances, each defining a specific operation to be performed upon a column. These include renames, type conversions, and custom computations. By default, this is an empty `list`. |
| `preserve_columns` | `List[str]` or `str` | Optional regular expression string or list of regular expression strings that define columns in the input data which should be preserved in the output `DataFrame`. By default, this is an empty `list`. |
| `row_filter` | `function` or `None` | Optional function to be called after all other processing has been performed. This function receives the `DataFrame` as its only argument returning a `DataFrame`. |

#### Column Info (`ColumnInfo`)
Defines a single column and type-cast.
| Argument | Type | Description |
| -------- | ---- | ----------- |
| `name` | `str` | Name of the column |
| `dtype` | `str` or Python type | Any type string or Python class recognized by [pandas](https://pandas.pydata.org/docs/user_guide/basics.html#dtypes) |

#### Custom Column (`CustomColumn`)
Subclass of `ColumnInfo`, defines a column to be computed by a user-defined function `process_column_fn`.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `name` | `str` | Name of the column |
| `dtype` | `str` or Python type | Any type string or Python class recognized by [pandas](https://pandas.pydata.org/docs/user_guide/basics.html#dtypes) |
| `process_column_fn` | `function` | Function which receives the entire `DataFrame` as its only input, returning a new [`pandas.Series`](https://pandas.pydata.org/docs/reference/api/pandas.Series.html) object to be stored in column `name`. |
| `input_column_types` | `dict[str, str]` | The input columns and the expected [`dtype` strings](https://pandas.pydata.org/docs/user_guide/basics.html#dtypes) that are needed for this Column to successfully process. Setting this as `None` will pass all columns. Specifying which columns are needed improves performance. |

#### Rename Column (`RenameColumn`)
Subclass of `ColumnInfo`, adds the ability to also perform a rename.
| Argument | Type | Description |
| -------- | ---- | ----------- |
| `name` | `str` | Name of the destination column |
| `dtype` | `str` or Python type | Any type string or Python class recognized by [pandas](https://pandas.pydata.org/docs/user_guide/basics.html#dtypes) |
| `input_name` | `str` | Original column name |

#### Boolean Column (`BoolColumn`)
Subclass of `RenameColumn`, adds the ability to map a set custom values as boolean values. For example say we had a string input field containing one of five possible `enum` values: `OK`, `SUCCESS`, `DENIED`, `CANCELED` and `EXPIRED` we could map these values into a single boolean field as:
```python
from morpheus.utils.column_info import BoolColumn
```
```python
field = BoolColumn(name="result",
                   dtype=bool,
                   input_name="result",
                   true_values=["OK", "SUCCESS"],
                   false_values=["DENIED", "CANCELED", "EXPIRED"])
```

We used strings in this example; however, we also could have just as easily mapped integer status codes. We also have the ability to map onto types other than boolean by providing custom values for true and false  (for example,  `1`/`0`, `yes`/`no`) .

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `name` | `str` | Name of the destination column |
| `dtype` | `str` or Python type | Typically this should be `bool` ; however, it could potentially be another type if `true_value` and `false_value` are specified. |
| `input_name` | `str` | Original column name |
| `true_value` | Any | Optional value to store for true values, should be of a type `dtype`. Defaults to `True`. |
| `false_value` | Any | Optional value to store for false values, should be of a type `dtype`. Defaults to `False`. |
| `true_values` | `List[str]` | List of string values to be interpreted as true. |
| `false_values` | `List[str]` | List of string values to be interpreted as false. |

#### Date-Time Column (`DateTimeColumn`)
Subclass of `RenameColumn`, specific to casting UTC localized `datetime` values. When incoming values contain a time-zone offset string the values are converted to UTC, while values without a time-zone are assumed to be UTC.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `name` | `str` | Name of the destination column |
| `dtype` | `str` or Python type | Any type string or Python class recognized by [pandas](https://pandas.pydata.org/docs/user_guide/basics.html#dtypes) |
| `input_name` | `str` | Original column name |

#### String-Join Column (`StringJoinColumn`)
Subclass of `RenameColumn`, converts incoming `list` values to string by joining by `sep`.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `name` | `str` | Name of the destination column |
| `dtype` | `str` or Python type | Any type string or Python class recognized by [pandas](https://pandas.pydata.org/docs/user_guide/basics.html#dtypes) |
| `input_name` | `str` | Original column name |
| `sep` | `str` | Separator string to use for the join |

#### String-Cat Column (`StringCatColumn`)
Subclass of `ColumnInfo`, concatenates values from multiple columns into a new string column separated by `sep`.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `name` | `str` | Name of the destination column |
| `dtype` | `str` or Python type | Any type string or Python class recognized by [pandas](https://pandas.pydata.org/docs/user_guide/basics.html#dtypes) |
| `input_columns` | `List[str]` | List of columns to concatenate |
| `sep` | `str` | Separator string |

#### Increment Column (`IncrementColumn`)
Subclass of `DateTimeColumn`, counts the unique occurrences of a value in `groupby_column` over a specific time window `period` based on dates in the `input_name` field.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `name` | `str` | Name of the destination column |
| `dtype` | `str` or Python type | Should be `int` or other integer class |
| `input_name` | `str` | Original column name containing timestamp values |
| `groupby_column` | `str` | Column name to group by |
| `period` | `str` | Optional time period to perform the calculation over, value must be [one of pandas' offset strings](https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases). Defaults to `D` one day |

### Input Stages
![Input Stages](img/dfp_input_config.png)

#### Source Stage (`MultiFileSource`)
The `MultiFileSource` (`python/morpheus/morpheus/modules/input/multi_file_source.py`) receives a path or list of paths (`filenames`), and will collectively be emitted into the pipeline as an [`fsspec.core.OpenFiles`](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.core.OpenFiles) object. The paths may include wildcards `*` as well as URLs (ex: `s3://path`) to remote storage providers such as S3, FTP, GCP, Azure, Databricks and others as defined by [`fsspec`](https://filesystem-spec.readthedocs.io/en/latest/api.html?highlight=open_files#fsspec.open_files). In addition to this paths can be cached locally by prefixing them with `filecache::` (ex: `filecache::s3://bucket-name/key-name`).

> **Note:**  This stage does not actually download the data files, allowing the file list to be filtered and batched prior to being downloaded.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `c` | `morpheus.config.Config` | Morpheus configuration object |
| `filenames` | `List[str]` or `str` | Paths to source file to be read from |
| `watch` | `bool` | Optional: when `True` will repeatedly poll `filenames` for new files. This assumes that at least one of the paths in `filenames` contains a wildcard. By default `False`. |
| `watch_interval` | `float` | When `watch` is `True`, this is the time in seconds between polling the paths in `filenames` for new files. Ignored when `watch` is `False`. |


#### File Batcher Stage (`DFPFileBatcherStage`)
The `DFPFileBatcherStage` (`python/morpheus_dfp/morpheus_dfp/stages/dfp_file_batcher_stage.py`) groups data in the incoming `DataFrame` in batches of a time period (per day default), and optionally filtering incoming data to a specific time window. This stage can potentially improve performance by combining multiple small files into a single batch. This stage assumes that the date of the logs can be easily inferred such as encoding the creation time in the file name (for example, `AUTH_LOG-2022-08-21T22.05.23Z.json`), or using the modification time as reported by the file system. The actual method for extracting the date is encoded in a user-supplied `date_conversion_func` function (more on this later).

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `c` | `morpheus.config.Config` | Morpheus configuration object |
| `date_conversion_func` | `function` | Function receives a single [`fsspec.core.OpenFile`](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.core.OpenFile) argument and returns a `datetime.datetime` object |
| `period` | `str` | Time period to group data by, value must be [one of pandas' offset strings](https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases) |
| `sampling_rate_s` | `int` | Optional, default=`None`. Deprecated consider using `sampling` instead. When defined a subset of the incoming data files will be sampled, taking the first row for each `sampling_rate_s` seconds.|
| `start_time` | `datetime` | Optional, default=`None`. When not None incoming data files will be filtered, excluding any files created prior to `start_time` |
| `end_time` | `datetime` | Optional, default=`None`. When not None incoming data files will be filtered, excluding any files created after `end_time` |
| `sampling` | `str`, `float`, `int` | Optional,  When non-None a subset of the incoming data files will be sampled. When a string, the value is interpreted as a pandas frequency. The first row for each frequency will be taken. When the value is between [0,1), a percentage of rows will be taken. When the value is greater than 1, the value is interpreted as the random count of rows to take. |

For situations where the creation date of the log file is encoded in the filename, the `date_extractor` in the `morpheus/utils/file_utils.py` module can be used. The `date_extractor` assumes that the timestamps are localized to UTC and will need to have a regex pattern bound to it before being passed in as a parameter to `DFPFileBatcherStage`. The regex pattern will need to contain the following named groups: `year`, `month`, `day`, `hour`, `minute`, `second`, and optionally `microsecond`. In cases where the regular expression does not match the `date_extractor` function will fallback to using the modified time of the file.

For input files containing an ISO 8601 formatted date string the `iso_date_regex` regex can be used ex:
```python
from functools import partial

from morpheus.utils.file_utils import date_extractor
from morpheus_dfp.utils.regex_utils import iso_date_regex
```
```python
# Batch files into buckets by time. Use the default ISO date extractor from the filename
pipeline.add_stage(
    DFPFileBatcherStage(config,
                        period="D",
                        date_conversion_func=functools.partial(date_extractor, filename_regex=iso_date_regex)))
```

> **Note:**  If `date_conversion_func` returns time-zone aware timestamps, then `start_time` and `end_time` if not `None` need to also be timezone aware `datetime` objects.

#### File to DataFrame Stage (`DFPFileToDataFrameStage`)
The `DFPFileToDataFrameStage` (`python/morpheus_dfp/morpheus_dfp/stages/dfp_file_to_df.py`) stage receives a `list` of an [`fsspec.core.OpenFiles`](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.core.OpenFiles) and loads them into a single `DataFrame` which is then emitted into the pipeline. When the parent stage is `DFPFileBatcherStage` each batch (typically one day) is concatenated into a single `DataFrame`. If the parent was `MultiFileSource` the entire dataset is loaded into a single `DataFrame`. Because of this, it is important to choose a `period` argument for `DFPFileBatcherStage` small enough such that each batch can fit into memory.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `c` | `morpheus.config.Config` | Morpheus configuration object |
| `schema` | `DataFrameInputSchema` | Schema specifying columns to load, along with any necessary renames and data type conversions  |
| `filter_null` | `bool` | Optional: Whether to filter null rows after loading, by default True. |
| `file_type` | `morpheus.common.FileTypes` (`enum`) | Optional: Indicates file type to be loaded. Currently supported values at time of writing are: `FileTypes.Auto`, `FileTypes.CSV`, `FileTypes.JSON` and `FileTypes.PARQUET`. Default value is `FileTypes.Auto` which will infer the type based on the file extension, set this value if using a custom extension |
| `parser_kwargs` | `dict` or `None` | Optional: additional keyword arguments to be passed into the `DataFrame` parser, currently this is going to be either [`pandas.read_csv`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html), [`pandas.read_json`](https://pandas.pydata.org/docs/reference/api/pandas.read_json.html) or [`pandas.read_parquet`](https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html) |
| `cache_dir` | `str` | Optional: path to cache location, defaults to `./.cache/dfp` |

This stage is able to download and load data files concurrently by multiple methods. Currently supported methods are: `single_thread`, `dask`, and `dask_thread`. The method used is chosen by setting the {envvar}`MORPHEUS_FILE_DOWNLOAD_TYPE` environment variable, and `dask_thread` is used by default, and `single_thread` effectively disables concurrent loading.

This stage will cache the resulting `DataFrame` in `cache_dir`, since we are caching the `DataFrame`s and not the source files, a cache hit avoids the cost of parsing the incoming data. In the case of remote storage systems, such as S3, this avoids both parsing and a download on a cache hit. One consequence of this is that any change to the `schema` will require purging cached files in the `cache_dir` before those changes are visible.

> **Note:**  This caching is in addition to any caching which may have occurred when using the optional `filecache::` prefix.

### Output Stages
![Output Stages](img/dfp_output_config.png)

For the inference pipeline, any Morpheus output stage, such as {py:obj}`~morpheus.stages.output.write_to_file_stage.WriteToFileStage` and {py:obj}`~morpheus.stages.output.write_to_kafka_stage.WriteToKafkaStage`, could be used in addition to the `WriteToS3Stage` documented below.

#### Write to File Stage (`WriteToFileStage`)
This final stage will write all received messages to a single output file in either CSV or JSON format.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `c` | `morpheus.config.Config` | Morpheus configuration object |
| `filename` | `str` | The file to write anomalous log messages to. |
| `overwrite` | `bool` | Optional, defaults to `False`. If the file specified in `filename` already exists, it will be overwritten if this option is set to `True` |

#### Write to S3 Stage (`WriteToS3Stage`)
The {py:obj}`~morpheus_dfp.stages.write_to_s3_stage.WriteToS3Stage` stage writes the resulting anomaly detections to S3. The `WriteToS3Stage` decouples the S3 specific operations from the Morpheus stage, and as such receives an `s3_writer` argument.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `c` | `morpheus.config.Config` | Morpheus configuration object |
| `s3_writer` | `function` | User defined function which receives an instance of a `morpheus.messages.message_meta.MessageMeta` and returns that same message instance. Any S3 specific configurations, such as bucket name, should be bound to the method. |

### Core Pipeline
These stages are common to both the training and inference pipelines, unlike the input and output stages these are specific to the DFP pipeline and intended to be configured but not replaceable.

#### Split Users Stage (`DFPSplitUsersStage`)
The {py:obj}`~morpheus_dfp.stages.dfp_split_users_stage.DFPSplitUsersStage` stage receives an incoming `DataFrame` and emits a `list` of `DFPMessageMeta` where each `DFPMessageMeta` represents the records associated for a given user. This allows for downstream stages to perform all necessary operations on a per user basis.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `c` | `morpheus.config.Config` | Morpheus configuration object |
| `include_generic` | `bool` | When `True` a `DFPMessageMeta` will be constructed for the generic user containing all records not excluded by the `skip_users` and `only_users` filters |
| `include_individual` | `bool` | When `True` a `DFPMessageMeta` instance will be constructed for each user not excluded by the `skip_users` and `only_users` filters |
| `skip_users` | `List[str]` or `None` | List of users to exclude, when `include_generic` is `True` excluded records will also be excluded from the generic user. Mutually exclusive with `only_users`. |
| `only_users` | `List[str]` or `None` | Limit records to a specific list of users, when `include_generic` is `True` the generic user's records will also be limited to the users in this list. Mutually exclusive with `skip_users`. |

#### Rolling Window Stage (`DFPRollingWindowStage`)
The {py:obj}`~morpheus_dfp.stages.dfp_rolling_window_stage.DFPRollingWindowStage` stage performs several key pieces of functionality for DFP.
<!-- Work-around for https://github.com/errata-ai/vale/issues/874 -->
<!-- vale off -->
1. This stage keeps a moving window of logs on a per user basis
   * These logs are saved to disk to reduce memory requirements between logs from the same user
1. It only emits logs when the window history requirements are met
   * Until all of the window history requirements are met, no messages will be sent to the rest of the pipeline.
   * Configuration options for defining the window history requirements are detailed below.
1. It repeats the necessary logs to properly calculate log dependent features.
   * To support all column feature types, incoming log messages can be combined with existing history and sent to downstream stages.
   * For example, to calculate a feature that increments a counter for the number of logs a particular user has generated in a single day, we must have the user's log history for the past 24 hours. To support this, this stage will combine new logs with existing history into a single `DataFrame`.
   * It is the responsibility of downstream stages to distinguish between new logs and existing history.
<!-- vale on -->

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `c` | `morpheus.config.Config` | Morpheus configuration object |
| `min_history` | `int` | Exclude users with less than `min_history` records, setting this to `1` effectively disables this feature |
| `min_increment` | `int` | Exclude incoming batches for users where less than `min_increment` new records have been added since the last batch, setting this to `0` effectively disables this feature |
| `max_history` | `int`, `str` or `None` | When not `None`, include up to `max_history` records. When `max_history` is an int, then the last `max_history` records will be included. When `max_history` is a `str` it is assumed to represent a duration parsable by [`pandas.Timedelta`](https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html) and only those records within the window of [latest timestamp - `max_history`, latest timestamp] will be included. |
| `cache_dir` | `str` | Optional path to cache directory, cached items will be stored in a subdirectory under `cache_dir` named `rolling-user-data` this directory, along with `cache_dir` will be created if it does not already exist. |


> **Note:**  this stage computes a row hash for the first and last rows of the incoming `DataFrame` as such all data contained must be hashable, any non-hashable values such as `lists` should be dropped or converted into hashable types in the `DFPFileToDataFrameStage`.

#### Preprocessing Stage (`DFPPreprocessingStage`)
The {py:obj}`~morpheus_dfp.stages.dfp_preprocessing_stage.DFPPreprocessingStage` stage, the actual logic of preprocessing is defined in the `input_schema` argument. Since this stage occurs in the pipeline after the `DFPFileBatcherStage` and `DFPSplitUsersStage` stages all records in the incoming `DataFrame` correspond to only a single user within a specific time period allowing for columns to be computer on a per-user per-time period basis such as the `logcount` and `locincrement` features mentioned above. Making the type of processing performed in this stage different from those performed in the `DFPFileToDataFrameStage`.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `c` | `morpheus.config.Config` | Morpheus configuration object |
| `input_schema` | `DataFrameInputSchema` | Schema specifying columns to be included in the output `DataFrame` including computed columns |

## Training Pipeline
![Training PipelineOverview](img/dfp_training_overview.png)

Training must begin with the generic user model which is trained with the logs from all users. This model serves as a fallback model for users and accounts without sufficient training data. The name of the generic user is defined in the `ae.fallback_username` attribute of the Morpheus configuration object and defaults to `generic_user`.

After training the generic model, individual user models can be trained. Individual user models provide better accuracy but require sufficient data. Many users do not have sufficient data to train the model accurately.

### Training Stages

#### Training Stage (`DFPTraining`)
The {py:obj}`~morpheus_dfp.stages.dfp_training.DFPTraining` trains a model for each incoming `DataFrame` and emits an instance of `morpheus.messages.ControlMessage` containing the trained model.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `c` | `morpheus.config.Config` | Morpheus configuration object |
| `model_kwargs` | `dict` or `None` | Optional dictionary of keyword arguments to be used when constructing the model. Refer to [`AutoEncoder`](https://github.com/nv-morpheus/dfencoder/blob/master/dfencoder/autoencoder.py) for information on the available options.|
| `epochs` | `int` | Number of training epochs. Default is 30.|
| `validation_size` | `float` | Proportion of the input dataset to use for training validation. Should be between 0.0 and 1.0. Default is 0.0.|

#### MLflow Model Writer Stage (`DFPMLFlowModelWriterStage`)
The {py:obj}`~morpheus_dfp.stages.dfp_mlflow_model_writer.DFPMLFlowModelWriterStage` stage publishes trained models into MLflow, skipping any model which lacked sufficient training data (current required minimum is 300 log records).

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `c` | `morpheus.config.Config` | Morpheus configuration object |
| `model_name_formatter` | `str` | Optional format string to control the name of models stored in MLflow, default is `dfp-{user_id}`. Currently available field names are: `user_id` and `user_md5` which is an md5 hexadecimal digest as returned by [`hash.hexdigest`](https://docs.python.org/3.10/library/hashlib.html?highlight=hexdigest#hashlib.hash.hexdigest). |
| `experiment_name_formatter` | `str` | Optional format string to control the experiment name for models stored in MLflow, default is `/dfp-models/{reg_model_name}`. Currently available field names are: `user_id`, `user_md5` and `reg_model_name` which is the model name as defined by `model_name_formatter` once the field names have been applied. |
| `databricks_permissions` | `dict` or `None` | Optional, when not `None` sets permissions needed when using a Databricks hosted MLflow server |

> **Note:**  If using a remote MLflow server, users will need to call [`mlflow.set_tracking_uri`](https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri) before starting the pipeline.

## Inference Pipeline
![Inference Pipeline Overview](img/dfp_inference_overview.png)

### Inference Stages

#### Inference Stage (`DFPInferenceStage`)
The {py:obj}`~morpheus_dfp.stages.dfp_inference_stage.DFPInferenceStage` stage loads models from MLflow and performs inferences against those models. This stage emits a message containing the original `DataFrame` along with new columns containing the z score (`mean_abs_z`), as well as the name and version of the model that generated that score (`model_version`). For each feature in the model, three additional columns will also be added:
* `<feature name>_loss` : The loss
* `<feature name>_z_loss` : The loss z-score
* `<feature name>_pred` : The predicted value

For a hypothetical feature named `result`, the three added columns will be: `result_loss`, `result_z_loss`, `result_pred`.

For performance models fetched from MLflow are cached locally and are cached for up to 10 minutes allowing updated models to be routinely updated. In addition to caching individual models, the stage also maintains a cache of which models are available, so a newly trained user model published to MLflow won't be visible to an already running inference pipeline for up to 10 minutes.

For any user without an associated model in MLflow, the model for the generic user is used. The name of the generic user is defined in the `ae.fallback_username` attribute of the Morpheus configuration object defaults to `generic_user`.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `c` | `morpheus.config.Config` | Morpheus configuration object |
| `model_name_formatter` | `str` | Format string to control the name of models fetched from MLflow. Currently available field names are: `user_id` and `user_md5` which is an md5 hexadecimal digest as returned by [`hash.hexdigest`](https://docs.python.org/3.10/library/hashlib.html?highlight=hexdigest#hashlib.hash.hexdigest). |

#### Filter Detection Stage (`FilterDetectionsStage`)
The {py:obj}`~morpheus.stages.postprocess.filter_detections_stage.FilterDetectionsStage` stage filters the output from the inference stage for any anomalous messages. Logs which exceed the specified Z-Score will be passed onto the next stage. All remaining logs which are below the threshold will be dropped. For the purposes of the DFP pipeline, this stage is configured to use the `mean_abs_z` column of the DataFrame as the filter criteria.

| Name | Type | Default | Description |
| --- | --- | --- | :-- |
| `threshold` | `float` | `0.5` | The threshold value above which logs are considered to be anomalous. The default is `0.5`; however, the DFP pipeline uses a value of `2.0`. All normal logs will be filtered out and anomalous logs will be passed on. |
| `copy` | `bool` | `True` | When the `copy` argument is `True` (default), rows that meet the filter criteria are copied into a new DataFrame. When `False` sliced views are used instead. This is a performance optimization, and has no functional impact. |
| `filter_source` | `FilterSource` | `FilterSource.Auto` | Indicates if the filter criteria exists in an output tensor (`FilterSource.TENSOR`) or a column in a DataFrame (`FilterSource.DATAFRAME`). |
| `field_name` | `str` | `probs` | Name of the tensor (`filter_source=FilterSource.TENSOR`) or DataFrame column (`filter_source=FilterSource.DATAFRAME`) to use as the filter criteria. |

#### Post Processing Stage (`DFPPostprocessingStage`)
The {py:obj}`~morpheus_dfp.stages.dfp_postprocessing_stage.DFPPostprocessingStage` stage adds a new `event_time` column to the DataFrame indicating the time which Morpheus detected the anomalous messages, and replaces any `NAN` values with the a string value of `'NaN'`.
