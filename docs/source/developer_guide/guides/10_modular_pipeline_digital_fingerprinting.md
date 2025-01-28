<!--
SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Introduction to Modular Digital Fingerprinting Pipeline (Integrated Training)

## Table of Contents

- [Introduction to Modular Digital Fingerprinting Pipeline (Integrated Training)](#introduction-to-modular-digital-fingerprinting-pipeline-integrated-training)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Motivation](#motivation)
    - [Overview](#overview)
  - [Setting up Morpheus](#setting-up-morpheus)
  - [Morpheus Modules](#morpheus-modules)
  - [DFP Deployment](#dfp-deployment)
    - [`fsspec` Data Loader](#fsspec-data-loader)
  - [DFP Training and Inference Pipelines](#dfp-training-and-inference-pipelines)
    - [DFP Preprocessing](#dfp-preprocessing)
    - [Control Message Filter](#control-message-filter)
    - [File Batcher](#file-batcher)
    - [File to DF DataLoader](#file-to-df-dataloader)
    - [DFP Split Users](#dfp-split-users)
    - [DFP Rolling Window](#dfp-rolling-window)
    - [DFP Data Prep](#dfp-data-prep)
  - [DFP Training Pipeline](#dfp-training-pipeline)
    - [DFP Training](#dfp-training)
    - [MLflow Model Writer](#mlflow-model-writer)
  - [DFP Inference Pipeline](#dfp-inference-pipeline)
    - [DFP Inference](#dfp-inference)
    - [Filter Detections](#filter-detections)
    - [DFP Post Processing](#dfp-post-processing)
    - [Serialize](#serialize)
    - [Write to File](#write-to-file)
  - [Running Example Modular DFP Pipelines](#running-example-modular-dfp-pipelines)
    - [System requirements](#system-requirements)
    - [Building the services](#building-the-services)
    - [Downloading the example datasets](#downloading-the-example-datasets)
    - [Run Morpheus pipeline](#run-morpheus-pipeline)
    - [Output Fields](#output-fields)

## Introduction


### Motivation


This document presents the adaptation of the Digital Fingerprinting pipeline in Morpheus from the existing stage-based approach to one that is module-based; this process will provide a basis to work through motivation and usage examples for number of new features found in the 23.03 release. The updated pipeline incorporates extensions to facilitate event-driven workflows and human-in-the-loop interactions through the use of control messages. Moreover, it introduces a dynamic method for acquiring and loading data, further enhancing the pipeline's capabilities.

The pipeline comprises a series of interconnected modules designed to create a versatile split processing pipeline. This design enables the reception and processing of control messages to perform tasks such as inference against observed events using either generic or user-specific models. Additionally, the pipeline can train new models based on aggregated or predefined training data, offering a more adaptable and user-friendly experience.

### Overview


At a high level, the pipeline consists of three parts: the front-end file list loader that reads new control messages from a Kafka topic and expands the described data sources to be processed, and the training and inference pipelines that process the data. The updated pipeline enables the reception and processing of control messages, allowing for tasks such as inference against observed events using generic or user-specific models. It also allows for the training of new models based on aggregated or predefined training data.

The front-end loader outputs one or more control messages that are passed to the training and inference pipelines. Messages are either dropped or passed to the next module in the pipeline based on the message type. The updated pipeline implicitly supports two distinct workflows: inference and training. However, because of the use of control messages, it can also support a hybrid data processing workflow that handles both streaming data processed in real-time, aggregated, batched, and subsequently used for training, as well as interleaved self-contained training tasks that specify the training data to be used and are able to bypass batching and aggregation stages.

Moreover, the updated pipeline supports human-in-the-loop workflows, such as the ability to manually trigger training or inference tasks against a specific set of data, and the capacity for real-time labeling of production inference events that can be injected back into the training pipeline.

The following content will track the pipeline declared in `examples/digital_fingerprinting/production/dfp_integrated_training_streaming_pipeline.py`

```python
# Setup and command line argument parsing
...

# Create an empty pipeline
pipeline = Pipeline(config)

# Create our Kafka source stage that we can read control messages from.
source_stage = pipeline.add_stage(
    ControlMessageKafkaSourceStage(config,
                                   bootstrap_servers=kwargs["bootstrap_servers"],
                                   input_topic=list(kwargs["input_topic"]),
                                   group_id=kwargs["group_id"],
                                   poll_interval=kwargs["poll_interval"],
                                   disable_commit=kwargs["disable_commit"],
                                   disable_pre_filtering=kwargs["disable_pre_filtering"]))

# Create our DFP deployment stage that will load our Digital Fingerprinting module into the pipeline.
dfp_deployment_stage = pipeline.add_stage(
    MultiPortModulesStage(config,
                          dfp_deployment_module_config,
                          input_ports=["input"],
                          output_ports=["output_0", "output_1"]))

# Connect the source stage to the DFP deployment module
pipeline.add_edge(source_stage, dfp_deployment_stage)

# Run the pipeline
pipeline.run()
```

## Setting up Morpheus

For a full introduction in how to set up and run Morpheus, please refer to the [Getting Started](../../getting_started.md) guide.

## Morpheus Modules

For a full introduction to Morpheus modules, refer to the [Python Modules](7_python_modules.md) and [C++ Modules](8_cpp_modules.md) guides.

## DFP Deployment


Source: `python/morpheus_dfp/morpheus_dfp/modules/dfp_deployment.py`

This is the top level module that encapsulates the entire Digital Fingerprinting pipeline, it is primarily responsible for wrapping the training and inference pipelines, providing the correct module interface, and doing some configuration pre-processing. Since this module is monolithic, it supports a significant number of configuration options; however, the majority of these have intelligent defaults and are not required to be specified.

The module consists of three chained sub-modules:

- `fsspec_dataloader`
    - Responsible for expanding data source declarations into individual files that can be processed by the pipeline.
- `dfp_training_pipe`
    - Connected to the output of the preprocessing stage. Responsible for handling training based control messages.
- `dfp_inference_pipe`
    - Connected to the output of the preprocessing stage. Responsible for handling inference based control messages.

For a complete reference, refer to: [DFP Deployment](../../modules/examples/digital_fingerprinting/dfp_deployment.md)

```python
@register_module(DFP_DEPLOYMENT, MORPHEUS_MODULE_NAMESPACE)
def dfp_deployment(builder: mrc.Builder):
    # Setup and configuration parsing
    ...

    # Make an edge between modules
    builder.make_edge(fsspec_dataloader_module.output_port("output"), router)
    builder.make_edge(router.get_source("training"), dfp_training_pipe_module.input_port("input"))
    builder.make_edge(router.get_source("inference"), dfp_inference_pipe_module.input_port("input"))

    out_nodes = [dfp_training_pipe_module.output_port("output"), dfp_inference_pipe_module.output_port("output")]

    # Register input port for a module.
    builder.register_module_input("input", fsspec_dataloader_module.input_port("input"))
```

### `fsspec` Data Loader


Source: `morpheus/loaders/fsspec_loader.py`

This is an instance of the new DataLoader module, utilizing a pre-defined `fsspec` style loader. The module is used to transform glob specified file lists into individual file paths and update the control message with those paths.

For a complete reference, refer to: [DataLoader Module](../../modules/core/data_loader.md)

## DFP Training and Inference Pipelines

There are a number of modules that are used in both the training and inference pipelines, but which are be configured independently. We'll introduce Shared modules here and then dive into the unique modules for each pipeline.

### DFP Preprocessing

Source: `python/morpheus_dfp/morpheus_dfp/modules/dfp_preproc.py`

The `dfp_preproc` module is a functional component within the Morpheus framework that combines multiple processing pipeline modules related to inference and training. This module simplifies the pipeline by consolidating various modules into a single, cohesive unit. The `dfp_preproc` module supports configuration parameters such as the cache directory, timestamp column name, batching options, user splitting options, and supported data loaders for various file types.

The module itself consists of a series of chained sub-modules, which are connected in a logical sequence:

- `file_batcher_module`
    - Responsible for batching files, either into a single control message in the case of an encapsulated training message, or into a series of control messages in the of streaming data.
- `file_to_df_dataloader_module`
    - Responsible for file retrieval and insertion into a cuDF DataFrame.
- `dfp_split_users_module`
    - Responsible for splitting the DataFrame into a series of DataFrames, one per user.

For a complete reference, refer to: [`dfp_preproc`](../../modules/examples/digital_fingerprinting/dfp_preproc.md)

### Control Message Filter

Source: `morpheus/modules/filter_control_message.py`

The `filter_control_message` module is a component designed to discard control messages based on specified filtering criteria. This module allows users to configure filtering options such as task type and data type. When either task filtering or data type filtering is enabled, the module processes control messages to verify if they meet the specified criteria. If the control message does not match the criteria, it is discarded. The module uses a node function to filter and process control messages, and registers input and output ports to facilitate seamless integration into the data processing pipeline.

For a complete reference, refer to: [Filter Control Message](../../modules/core/filter_control_message.md)

### File Batcher

Source: `morpheus/modules/file_batcher.py`

The `file_batcher` module is a component that is responsible for loading input files, filtering out files older than the specified time window, and grouping the remaining files by periods that fall within the time window. This module offers configuration for parameters such as batching options, cache directory, file type, filtering null values, data schema, and the timestamp column name. The `file_batcher` module processes control messages, validates them, and generates a list of files with their timestamps. The module then groups files by the given period, creates control messages for each batch, and sends them downstream for further processing. A node function is used to handle the processing of control messages, and input and output ports are registered to integrate the module into the data processing pipeline seamlessly.

The file batcher is one of the first pipeline components that begins to differ more substantially from the previous raw-data pipeline, prior to 23.03. In addition to its previous functionality, the file batcher is now control message aware, and can handle both streaming and encapsulated control messages, a property denoted by the `data_type` property of the control message's metadata being set as either `streaming` or `payload`. Additionally, the file batcher's default processing criteria for `period`, `sampling_rate_s`, `start_time`, and `end_time` can now be overridden by their corresponding values in the control message's `batching_options` metadata entry.

In the case of streaming data, the file batcher will operate as it did previously, grouping files as specified by the `period`, `sampling_rate_s`, `start_time`, and `end_time` properties, creating a control message for each batch, and forwarding them downstream. In the case of encapsulated data, the file batcher will operate similarly, but will only create a single control message for the entire payload, and forward it downstream. In this way, it is possible to attach all the necessary training data to a given training task, and skip any downstream aggregation.

For a complete reference, refer to: [File Batcher](../../modules/core/file_batcher.md)

### File to DF DataLoader

Source: `morpheus/loaders/file_to_df_loader.py`

This is an instance of the new DataLoader module, utilizing a pre-defined `file_to_df` style loader. The module is used to process `load` tasks that reference files which need to be retrieved, possibly cached, and then loaded into a cuDF DataFrame with is set as the control message payload.

For a complete reference, refer to: [DataLoader Module](../../modules/core/data_loader.md)

### DFP Split Users

Source: `python/morpheus_dfp/morpheus_dfp/modules/dfp_split_users.py`

The `dfp_split_users` module is responsible for splitting the input data based on user IDs. The module provides configuration options, such as fallback username, include generic user, include individual users, and specify lists of user IDs to include or exclude in the output.

The module processes control messages by extracting the user information from the message payload, filtering the data based on the provided configuration, and splitting the data by user ID. For each user ID, the function generates a new control message containing the corresponding data and sends it downstream for further processing.

For a complete reference, refer to: [DFP Split Users](../../modules/examples/digital_fingerprinting/dfp_split_users.md)


### DFP Rolling Window

Source: `python/morpheus_dfp/morpheus_dfp/modules/dfp_rolling_window.py`

The `dfp_rolling_window` module is responsible for maintaining a rolling window of historical data, acting as a streaming caching and batching system. The module provides various configuration options, such as aggregation span, cache directory, caching options, timestamp column name, and trigger conditions.

The main functionality of the module is to processes control messages containing data. For each control message, the function determines the user ID and data type, then tries to build a rolling window with the historical data from the cache. If enough data is available based on the trigger conditions, the function returns a control message with the appropriate historical data for further processing.

The Rolling Window module is another example of a module that has been updated to be control message aware. In that it will differentiate between streaming and payload control messages, and handle them accordingly. In the case of a streaming control message, the module will process the message as it did previously, and either cache the streaming data and return, or if the trigger conditions are met, return a control message with the appropriate historical data. In the case of a payload control message, the rolling window module will be skipped entirely and simply forward the message to the next stage.

The Rolling window module has also been updated to support an additional `batch` mode of operation, in which it will cache streaming data until the trigger conditions are met, generate a new control message with the all existing data, flush the cache, and then forward the message downstream. Batch caching is the default mode for the streaming inference pipeline, and improves performance by reducing the bookkeeping required. This mode of operation is denoted by the `cache_mode` property of the module's configuration.

For a complete reference, refer to: [DFP Rolling Window](../../modules/examples/digital_fingerprinting/dfp_rolling_window.md)


### DFP Data Prep

Source: `python/morpheus_dfp/morpheus_dfp/modules/dfp_data_prep.py`

The `dfp_data_prep` module is responsible for preparing data for either inference or model training. The module requires a defined schema for data preparation.

The main functionality of the module is in the `process_features` function. For each control message containing data, the function processes the columns of the data according to the given schema. The processed DataFrame is then applied to the control message payload.

For a complete reference, refer to: [DFP Data Prep](../../modules/examples/digital_fingerprinting/dfp_data_prep.md)

## DFP Training Pipeline

Source: `python/morpheus_dfp/morpheus_dfp/modules/dfp_training_pipe.py`

The DFP Training Pipe module is a consolidated module that integrates several DFP pipeline modules that are essential to the training process. This module function provides a single entry point to the training pipeline, simplifying the process of training a model. The module offers configurable parameters for various stages in the pipeline, including data batching, data preprocessing, and data encoding for model training. Additionally, the MLflow model writer options allow for the trained model to be saved for future use.

The module itself consists of a series of chained sub-modules, each of which performs a specific task in the training:

- `preproc`
    - Data filtering and preprocessing
- `dfp_rolling_window`
    - Data caching and batching
- `dfp_data_prep`
    - Data encoding
- `dfp_training`
    - Model training
- `mlflow_model_writer`
    - Model and telemetry saving to MLflow

For a complete reference, refer to: [DFP Training Pipe](../../modules/examples/digital_fingerprinting/dfp_training_pipe.md)

```python
@register_module(DFP_TRAINING_PIPE, MORPHEUS_MODULE_NAMESPACE)
def dfp_training_pipe(builder: mrc.Builder):
    # Setup and config parsing
    ...

    # Make an edge between the modules.
    builder.make_edge(file_batcher_module.output_port("output"), file_to_df_dataloader_module.input_port("input"))
    builder.make_edge(file_to_df_dataloader_module.output_port("output"), file_to_df_monitor_module.input_port("input"))
    builder.make_edge(file_to_df_monitor_module.output_port("output"), dfp_split_users_module.input_port("input"))
    builder.make_edge(dfp_split_users_module.output_port("output"), dfp_split_users_monitor_module.input_port("input"))

    # Register input and output port for a module.
    builder.register_module_input("input", file_batcher_module.input_port("input"))
    builder.register_module_output("output", dfp_split_users_monitor_module.output_port("output"))
```

### DFP Training

Source: `python/morpheus_dfp/morpheus_dfp/modules/dfp_training.py`

The `dfp_training` module function is responsible for training the model. The `on_data` function is defined to handle incoming `ControlMessage` instances. It retrieves the user ID and the input data from the `ControlMessage`, creates an instance of the `AutoEncoder` class with the specified `model_kwargs`, and trains the model on the input data. The output message includes the trained model and metadata.

For a complete reference, refer to: [DFP Training](../../modules/examples/digital_fingerprinting/dfp_training.md)


### MLflow Model Writer

Source: `morpheus/modules/mlflow_model_writer.py`

The `mlflow_model_writer` module is responsible for uploading trained models to the MLflow server.

For each `ControlMessage` received, containing a trained model, the function uploads the model to MLflow along with associated metadata such as experiment name, run name, parameters, metrics, and the model signature. If the MLflow server is running on Databricks, the function also applies the required permissions to the registered model.

For a complete reference, refer to: [MLflow Model Writer](../../modules/core/mlflow_model_writer.md)


## DFP Inference Pipeline

Source: `python/morpheus_dfp/morpheus_dfp/modules/dfp_inference_pipe.py`

The `dfp_inference_pipe` module function consolidates multiple digital fingerprinting pipeline (DFP) modules relevant to the inference process into a single module. Its purpose is to simplify the creation and configuration of an inference pipeline by combining all necessary components.

The module sets up a series of interconnected components that handle various stages of the inference process, such as preprocessing, rolling window aggregation, data preparation, inference, detection filtering, post-processing, serialization, and writing the output to a file.

The module itself consists of a series of chained sub-modules, each of which performs a specific task in the inference pipeline:

- `dfp_preproc`
    - Data filtering and preprocessing
- `dfp_rolling_window`
    - Data caching and batching
- `dfp_data_prep`
    - Data encoding
- `dfp_inference`
    - Model inference
- `filter_detections`
    - Detection filtering
- `dfp_post_proc`
    - Detection post-processing
- `serialize`
    - Detection serialization
- `write_to_file`
    - Detection writing to file

For a complete reference, refer to: [DFP Inference Pipe](../../modules/examples/digital_fingerprinting/dfp_inference_pipe.md)

```python
@register_module(DFP_INFERENCE_PIPE, MORPHEUS_MODULE_NAMESPACE)
def dfp_inference_pipe(builder: mrc.Builder):
    # Setup and config parsing
    ...

    # Make an edge between the modules.
    builder.make_edge(preproc_module.output_port("output"), dfp_rolling_window_module.input_port("input"))
    builder.make_edge(dfp_rolling_window_module.output_port("output"), dfp_data_prep_module.input_port("input"))
    builder.make_edge(dfp_data_prep_module.output_port("output"), dfp_inference_module.input_port("input"))
    builder.make_edge(dfp_inference_module.output_port("output"), filter_detections_module.input_port("input"))
    builder.make_edge(filter_detections_module.output_port("output"), dfp_post_proc_module.input_port("input"))
    builder.make_edge(dfp_post_proc_module.output_port("output"), serialize_module.input_port("input"))
    builder.make_edge(serialize_module.output_port("output"), write_to_file_module.input_port("input"))

    # Register input and output port for a module.
    builder.register_module_input("input", preproc_module.input_port("input"))
    builder.register_module_output("output", write_to_file_module.output_port("output"))
```

### DFP Inference

Source: `python/morpheus_dfp/morpheus_dfp/modules/dfp_inference.py`

The `dfp_inference` module function creates an inference module that retrieves trained models and performs inference on the input data. The module requires a `model_name_formatter` and a `fallback_username` to be configured in its parameters.

The function defines a `get_model` method to load the model for a specific user, and a `process_task` method to handle individual inference tasks. The `process_task` method retrieves the user ID, extracts the payload, and converts the `DataFrame` to pandas format. It then attempts to load the model for the specified user ID and perform inference using the loaded model. Finally, it adds any other columns from the input data to the results `DataFrame` and creates a message with the output and metadata.

For a complete reference, refer to: [DFP Inference](../../modules/examples/digital_fingerprinting/dfp_inference.md)

### Filter Detections

Source: `morpheus/modules/filter_detections.py`

The `filter_detections` module function is designed to filter rows from a `DataFrame` based on values in a tensor or `DataFrame` column according to a specified threshold. Rows are excluded if their associated value in the specified field is less than or equal to the threshold.

This module can operate in two modes, set by the copy argument. When `copy=True`, rows that meet the filter criteria are copied into a new `DataFrame`. When `copy=False`, sliced views are used instead.

The function defines the `find_detections` method to determine the filter source and identify the rows that match the filter criteria. The `filter_copy` and `filter_slice` methods are responsible for handling the filtering process based on the chosen mode.


For a complete reference, refer to: [Filter Detections](../../modules/core/filter_detections.md)

### DFP Post Processing

Source: `python/morpheus_dfp/morpheus_dfp/modules/dfp_postprocessing.py`

The `dfp_postprocessing` module function performs post-processing tasks on the input data.


For a complete reference, refer to: [DFP Post Processing](../../modules/examples/digital_fingerprinting/dfp_postprocessing.md)

### Serialize

Source: `morpheus/modules/serialize.py`

The serialize module function is responsible for filtering columns from a `ControlMessage` object and emitting a `MessageMeta` object.

The `convert_to_df` function converts a DataFrame to JSON lines. It takes a `ControlMessage` instance, `include_columns` (a pattern for columns to include), `exclude_columns` (a list of patterns for columns to exclude), and `columns` (a list of columns to include). The function filters the columns of the input DataFrame based on the include and exclude patterns and retrieves the metadata of the filtered columns.

The module function compiles the include and exclude patterns into regular expressions. It then creates a node using the `convert_to_df` function with the compiled include and exclude patterns and the specified columns.


For a complete reference, refer to: [Serialize](../../modules/core/serialize.md)

### Write to File

Source: `morpheus/modules/write_to_file.py`

The `write_to_file` module function writes all messages to a file.

The `convert_to_strings` function takes a DataFrame (either pandas or cuDF) and converts it into the appropriate string format based on the file type (JSON or CSV). It checks whether to include the index column or not.


For a complete reference, refer to: [Write to File](../../modules/core/write_to_file.md)

## Running Example Modular DFP Pipelines

The following are steps to run modular DFP pipelines with example Azure and Duo datasets.

### System requirements
* [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/) installed on the host machine​
* Supported GPU with [NVIDIA Container Toolkit​](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

> **Note:**  For GPU requirements refer to the [Getting Started](../../getting_started.md#requirements) guide.

### Building the services
From the root of the Morpheus repo, run:
```bash
cd examples/digital_fingerprinting/production
docker compose build
```

> **Note:** This requires version 1.28.0 or higher of Docker Compose, and preferably v2. If you encounter an error similar to:
>
> ```
> ERROR: The Compose file './docker-compose.yml' is invalid because:
> services.jupyter.deploy.resources.reservations value Additional properties are not allowed ('devices' was
> unexpected)
> ```
>
> This is most likely due to using an older version of the `docker-compose` command, instead re-run the build with `docker compose`. Refer to [Migrate to Compose V2](https://docs.docker.com/compose/migrate/) for more information.

### Downloading the example datasets
First, we will need to install `s3fs` and then run the `examples/digital_fingerprinting/fetch_example_data.py` script. This will download the example data into the `examples/data/dfp` dir.

From the Morpheus repo, run:
```bash
pip install s3fs
python examples/digital_fingerprinting/fetch_example_data.py all
```

### Run Morpheus pipeline
From the `examples/digital_fingerprinting/production` dir, run:
```bash
docker compose run morpheus_pipeline bash
```
To run the DFP pipelines with the example datasets within the container, run the following from `examples/digital_fingerprinting/production/`:

* Duo Training Pipeline
    ```bash
    python dfp_integrated_training_batch_pipeline.py \
        --log_level DEBUG \
        --source duo \
        --start_time "2022-08-01" \
        --duration "60d" \
        --train_users generic \
        --input_file "./morpheus/control_messages/duo_payload_training.json"
    ```

* Duo Inference Pipeline
    ```bash
    python dfp_integrated_training_batch_pipeline.py \
        --log_level DEBUG \
        --source duo \
        --start_time "2022-08-30" \
        --input_file "./morpheus/control_messages/duo_payload_inference.json"
    ```

* Duo Training + Inference Pipeline
    ```bash
    python dfp_integrated_training_batch_pipeline.py \
        --log_level DEBUG \
        --source duo \
        --start_time "2022-08-01" \
        --duration "60d" \
        --train_users generic \
        --input_file "./morpheus/control_messages/duo_payload_load_train_inference.json"
    ```

* Azure Training Pipeline
    ```bash
    python dfp_integrated_training_batch_pipeline.py \
        --log_level DEBUG \
        --source azure \
        --start_time "2022-08-01" \
        --duration "60d" \
        --train_users generic \
        --input_file "./morpheus/control_messages/azure_payload_training.json"
    ```

* Azure Inference Pipeline
    ```bash
    python dfp_integrated_training_batch_pipeline.py \
        --log_level DEBUG \
        --source azure \
        --start_time "2022-08-30" \
        --input_file "./morpheus/control_messages/azure_payload_inference.json"
    ```

* Azure Training + Inference Pipeline
    ```bash
    python dfp_integrated_training_batch_pipeline.py \
        --log_level DEBUG \
        --source azure \
        --start_time "2022-08-01" \
        --duration "60d" \
        --train_users generic \
        --input_file "./morpheus/control_messages/azure_payload_load_train_inference.json"
    ```

### Output Fields
The output files, `dfp_detectiions_duo.csv` and `dfp_detections_azure.csv`, will contain those logs from the input dataset for which an anomaly was detected; this is determined by the z-score in the `mean_abs_z` field. By default, any logs with a z-score of 2.0 or higher are considered anomalous. Refer to [`DFPPostprocessingStage`](6_digital_fingerprinting_reference.md#post-processing-stage-dfppostprocessingstage).

Most of the fields in the output files generated by running the above examples are input fields or derived from input fields. The additional output fields are:
| Field | Type | Description |
| ----- | ---- | ----------- |
| `event_time` | TEXT | ISO 8601 formatted date string, the time the anomaly was detected by Morpheus |
| `model_version` | TEXT | Name and version of the model used to performed the inference, in the form of `<model name>:<version>` |
| `max_abs_z` | FLOAT | Max z-score across all features |
| `mean_abs_z` | FLOAT | Average z-score across all features |

In addition to this, for each input feature the following output fields will exist:
| Field | Type | Description |
| ----- | ---- | ----------- |
| `<feature name>_loss` | FLOAT | The loss |
| `<feature name>_z_loss` | FLOAT | The loss z-score |
| `<feature name>_pred` | FLOAT | The predicted value |

Refer to [DFPInferenceStage](6_digital_fingerprinting_reference.md#inference-stage-dfpinferencestage) for more on these fields.
