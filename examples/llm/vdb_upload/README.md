<!--
SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Vector DataBase Upload (VDB Upload) Pipeline

## Table of Contents

1. [Background Information](#Background-Information)
    - [Purpose](#Purpose)
    - [Source Documents](#Source-Documents)
    - [Embedding Model](#Embedding-Model)
    - [Vector Database Service](#Vector-Database-Service)
2. [Implementation and Design Decisions](#Implementation-and-Design-Decisions)
3. [Getting Started](#Getting-Started)
    - [Prerequisites](#Prerequisites)
        - [Milvus Service](#Milvus-Service)
        - [Triton Service](#Triton-Service)
    - [Running the Morpheus Pipeline](#Running-the-Morpheus-Pipeline)
    - [Options for vdb_upload Command](#Options-for-vdb_upload-Command)
    - [Exporting and Deploying a Different Model from Huggingface](#Exporting-and-Deploying-a-Different-Model-from-Huggingface)

## Background Information

### Purpose

The primary objective of this example is to demonstrate the construction of a pipeline that performs the following
tasks:

- Accepts a collection of documents.
- Divides the documents into smaller segments or chunks.
- Computes the embedding vector for each of these chunks.
- Uploads the text chunks along with their associated embeddings to a Vector Database (VDB).

### Source Documents

- The pipeline is designed to process text-based input from various document types. Possible use cases could
  include structured documents like PDFs, dynamic sources such as web pages, and image-based documents through future
  Optical Character Recognition (OCR) integration.

- For this demonstration, the source documents are obtained from RSS feeds combined with a web scraper. The rationale
  behind this selection includes:
    - Emulating practical cyber scenarios: Cybersecurity RSS feeds can serve as the foundation for a comprehensive
      knowledge database, such as for a security chatbot.
    - Minimizing external dependencies: Relying on RSS feeds and web scraping avoids the need for specialized datasets
      or API keys.

### Embedding Model

- The pipeline can accommodate various embedding models that transform text into vectors of floating-point numbers.
  Several models from Huggingface, such as `paraphrase-multilingual-mpnet-base-v2`, `e5-large-v2`,
  and `all-mpnet-base-v2`, have been evaluated for compatibility.

- For the purposes of this demonstration, the model `all-MiniLM-L6-v2` is employed. This model is included via LFS
  in this repository, and was chosen for its efficiency and compactness, characterized by a smaller embedding dimension
  of 384.

### Vector Database Service

- The architecture is agnostic to the choice of Vector Database (VDB) for storing embeddings and their metadata. While
  the present implementation employs Milvus due to its GPU-accelerated indices, the design supports easy integration
  with other databases like Chroma or FAISS, should the need arise.

## Implementation and Design Decisions

### Implementation Details

[Original GitHub issue](https://github.com/nv-morpheus/Morpheus/issues/1298)

The pipeline is composed of three primary components:

1. **Document Source Handler**: This component is responsible for acquiring and preprocessing the text data. Given that
   we are using RSS feeds and a web scraper in this example, the handler's function is to fetch the latest updates from
   the feeds, perform preliminary data cleaning, and standardize the format for subsequent steps.

2. **Embedding Generator**: This is the heart of the pipeline, which takes the preprocessed text chunks and computes
   their embeddings. Leveraging the model `all-MiniLM-L6-v2` from Huggingface, the text data is transformed into
   embeddings with a dimension of 384.

3. **Vector Database Uploader**: Post embedding generation, this module takes the embeddings alongside their associated
   metadata and pushes them to a Vector Database (VDB). For our implementation, Milvus, a GPU-accelerated vector
   database, has been chosen.

### Rationale Behind Design Decisions

The selection of specific components and models was influenced by several factors:

- **Document Source Choice**: RSS feeds and web scraping offer a dynamic and continuously updating source of data. For
  the use-case of building a repository for a cybersecurity, real-time information fetching is a reasonable choice.

- **Model Selection for Embeddings**: `all-MiniLM-L6-v2` was chosen due to its efficiency in generating embeddings. Its
  smaller dimension ensures quick computations without compromising the quality of embeddings.

- **Vector Database**: For the purposes of this pipeline, Milvus was chosen due to its popularity, ease of use, and
  availability.

## Getting Started

### Prerequisites

Before running the pipeline, we need to ensure that the following services are running:

#### Ensure LFS files are downloaded

To retrieve models from LFS run the following:

```bash
./scripts/fetch_data.py fetch models
```

#### Milvus Service

- Follow the instructions [here](https://milvus.io/docs/install_standalone-docker.md) to install and run a Milvus
  service.

#### Triton Service

- Pull the Docker image for Triton:
  ```bash
  docker pull nvcr.io/nvidia/tritonserver:24.04-py3
  ```

- From the Morpheus repo root directory, run the following to launch Triton and load the `all-MiniLM-L6-v2` model:
  ```bash
  docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 -v $PWD/models:/models nvcr.io/nvidia/tritonserver:24.04-py3 tritonserver --model-repository=/models/triton-model-repo --exit-on-error=false --model-control-mode=explicit --load-model all-MiniLM-L6-v2
  ```

  This will launch Triton and only load the `all-MiniLM-L6-v2` model. Once Triton has loaded the model, the following
  will be displayed:
    ```
    +------------------+---------+--------+
    | Model            | Version | Status |
    +------------------+---------+--------+
    | all-MiniLM-L6-v2 | 1       | READY  |
    +------------------+---------+--------+
    ```

### Running the Morpheus Pipeline

The top-level entry point for each of the LLM example pipelines is examples/llm/main.py. This script accepts a set of
options and a pipeline to run. For the purposes of this document, we'll focus on the vdb_upload pipeline option, which
incorporates various functionalities like handling RSS and filesystem sources, embedding configurations, and vector
database (VDB) settings.

#### Configuration Balance Considerations

When configuring the Morpheus Pipeline, especially for stages like the RSS source and the Vector Database Upload, it's
important to balance responsiveness and performance.

- **RSS Source Stage**: The RSS source stage is responsible for yielding webpage links for processing. A larger batch size
  at this stage can lead to decreased responsiveness, as the subsequent web scraper stage may take a considerable amount of
  time to retrieve and process all the items in each batch. To ensure a responsive experience for users, it's recommended
  to configure the RSS source stage with a relatively smaller batch size. This adjustment tends to have minimal impact on
  overall performance while significantly improving the time to process each batch of links.

- **Vector Database Upload Stage**: At the other end of the pipeline, the Vector Database Upload stage has its own
  considerations. This stage experiences a significant transaction overhead. To mitigate this, it is advisable to configure
  this stage with the largest batch size possible. This approach helps in efficiently managing transaction overheads and
  improves the throughput of the pipeline, especially when dealing with large volumes of data.

Balancing these configurations ensures that the pipeline runs efficiently, with optimized responsiveness at the RSS
source stage and improved throughput at the Vector Database Upload stage.

### Run example:

Default example usage, with pre-defined RSS source

```bash
python examples/llm/main.py vdb_upload pipeline \
  --enable_cache \
  --enable_monitors \
  --embedding_model_name all-MiniLM-L6-v2
```

Usage with CLI-Defined Sources:

*Example: Defining an RSS Source via CLI*

```bash
python examples/llm/main.py vdb_upload pipeline \
  --source_type rss \
  --interval_secs 300 \
  --rss_request_timeout_sec 5.0 \
  --enable_cache \
  --enable_monitors \
  --embedding_model_name all-MiniLM-L6-v2
```

*Example: Defining a Filesystem Source via CLI*

```bash
python examples/llm/main.py vdb_upload pipeline \
  --source_type filesystem \
  --file_source "./morpheus/data/*" \
  --enable_monitors \
  --embedding_model_name all-MiniLM-L6-v2
```

*Example: Combining RSS and Filesystem Sources via CLI*

```bash
python examples/llm/main.py vdb_upload pipeline \
  --source_type rss --source_type filesystem \
  --file_source "./morpheus/data/*" \
  --interval_secs 600 \
  --enable_cache \
  --enable_monitors \
  --embedding_model_name all-MiniLM-L6-v2
```

*Example: Defining sources via a config file*
Note: see `vdb_config.yaml` for a full configuration example.

`vdb_config.yaml`

```yaml
vdb_pipeline:
  sources:
    - type: filesystem
      name: "demo_filesystem_source"
      config:
        batch_size: 1024
        enable_monitor: False
        extractor_config:
          chunk_size: 512
          chunk_overlap: 50
          num_threads: 10 # Number of threads to use for file reads
        filenames:
          - "/path/to/data/*"
        watch: false
```

*Example: Defining a custom source via a config file*
Note: See `vdb_config.yaml` for a full configuration example.
Note: This example uses the same module and config as the filesystem source example above, but explicitly specifies the
module to load

`vdb_config.yaml`

```yaml
vdb_pipeline:
  sources:
    - type: custom
      name: "demo_custom_filesystem_source"
      module_id: "file_source_pipe"  # Required for custom source, defines the source module to load
      module_output_id: "output"  # Required for custom source, defines the output of the module to use
      namespace: "morpheus_examples_llm"  # Required for custom source, defines the namespace of the module to load
      config:
        batch_size: 1024
        extractor_config:
          chunk_size: 512
          num_threads: 10  # Number of threads to use for file reads
        config_name_mapping: "file_source_config"
        filenames:
          - "/path/to/data/*"
        watch: false
```

```bash
python examples/llm/main.py vdb_upload pipeline \
  --vdb_config_path "./vdb_config.yaml"
```

## Morpheus Pipeline Configuration Schema

The Morpheus Pipeline configuration allows for detailed specification of various pipeline stages, including source
definitions (like RSS feeds and filesystem paths), embedding configurations, and vector database settings.

### Sources Configuration

The `sources` section allows you to define multiple data sources of different types: RSS, filesystem, and custom.

### Embeddings Configuration

- **isolate_embeddings**: Boolean to isolate embeddings.
- **model_kwargs**:
    - **force_convert_inputs**: Boolean to force the conversion of inputs.
    - **model_name**: Name of the model, e.g., `"all-MiniLM-L6-v2"`.
    - **server_url**: URL of the server, e.g., `"http://localhost:8001"`.
    - **use_shared_memory**: Boolean to use shared memory.

### Pipeline Configuration

- **edge_buffer_size**: Size of the edge buffer, e.g., `128`.
- **feature_length**: Length of the features, e.g., `512`.
- **max_batch_size**: Maximum size of the batch, e.g., `256`.
- **num_threads**: Number of threads, e.g., `10`.
- **pipeline_batch_size**: Size of the batch for the pipeline, e.g., `1024`.

#### RSS Source Configuration

- **type**: `'rss'`
- **name**: Name of the RSS source.
- **config**:
    - **batch_size**: Number of RSS feeds to process at a time.
    - **cache_dir**: Directory for caching.
    - **cooldown_interval_sec**: Cooldown interval in seconds.
    - **enable_cache**: Boolean to enable caching.
    - **enable_monitor**: Boolean to enable monitoring.
    - **feed_input**: List of RSS feed URLs.
    - **interval_sec**: Interval in seconds for fetching new feed items.
    - **request_timeout_sec**: Timeout in seconds for RSS feed requests.
    - **run_indefinitely**: Boolean to indicate continuous running.
    - **stop_after**: Stop after emitting a specific number of records.
    - **web_scraper_config**:
        - **chunk_overlap**: Overlap size for chunks.
        - **chunk_size**: Size of content chunks for processing.
        - **enable_cache**: Boolean to enable caching.

#### Filesystem Source Configuration

- **type**: `'filesystem'`
- **name**: Name of the filesystem source.
- **config**:
    - **batch_size**: Number of files to process at a time.
    - **chunk_overlap**: Overlap size for chunks.
    - **chunk_size**: Size of chunks for processing.
    - **converters_meta**: Metadata for converters.
        - **csv**:
            - **chunk_size**: Chunk size for CSV processing.
            - **text_column_names**: Column names to be used as text.
              - **column_name_0** Column name 0.
              - **column_name_1** Column name 1.
    - **enable_monitor**: Boolean to enable monitoring.
    - **extractor_config**:
        - **chunk_size**: Size of chunks for the extractor.
        - **num_threads**: Number of threads for file reads.
    - **filenames**: List of file paths to be processed.
    - **watch**: Boolean to watch for file changes.

#### Custom Source Configuration

- **type**: `'custom'`
- **name**: Name of the custom source.
- **config**:
    - **config_name_mapping**: Mapping name for file source config.
    - **module_id**: Identifier of the module to use.
    - **module_output_id**: Output identifier of the module.
    - **namespace**: Namespace of the module.
    - **other_config_parameter_1**: Other config parameter 1.
    - **other_config_parameter_2**: Other config parameter 2.

### Tokenizer Configuration

- **model_kwargs**:
    - **add_special_tokens**: Boolean to add special tokens.
    - **column**: Column name, e.g., `"content"`.
    - **do_lower_case**: Boolean to convert to lowercase.
    - **truncation**: Boolean to truncate.
    - **vocab_hash_file**: Path to the vocabulary hash file.
- **model_name**: Name of the tokenizer model.

### Vector Database (VDB) Configuration

- **embedding_size**: Size of the embeddings to store in the vector database.
- **recreate**: Boolean to recreate the resource if it exists.
- **resource_name**: Identifier for the resource in the vector database.
- **service**: Type of vector database service (e.g., `"milvus"`).
- **uri**: URI for connecting to the Vector Database server.

## Options for `vdb_upload` Command

The `vdb_upload` command has its own set of options and commands:

### Commands:

- `export-triton-model`
- `langchain`
- `pipeline`

### Exporting and Deploying a Different Model from Huggingface

If you're looking to incorporate a different embedding model from Huggingface into the pipeline, follow the steps below
using `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` as an example:

1. **Identify the Desired Model**:
    - Head over to the [Huggingface Model Hub](https://huggingface.co/models) and search for the model you want. For
      this example, we are looking at `e5-large-v2`.

2. **Run the Pipeline Call with the Chosen Model**:
    - Execute the following command with the model name you've identified:
      ```bash
      python examples/llm/main.py vdb_upload export-triton-model  --model_name \
       sentence-transformers/paraphrase-multilingual-mpnet-base-v2 --triton_repo ./models/triton-model-repo
      ```

3. **Handling Unauthorized Errors**:
    - Please ensure you provide the correct model name. A common pitfall is encountering an `unauthorized error`. If
      you see the following error:
      ```text
      requests.exceptions.HTTPError: 401 Client Error: Unauthorized for url:
      ```
      This typically means the model name you provided does not match the one available on Huggingface. Double-check
      the model name and try again.

4. **Confirm Successful Model Export**:
    - After running the command, ensure that the specified `--triton_repo` directory now contains the exported model in
      the correct format, ready for deployment.
    ```bash
    $ ls ${MORPHEUS_ROOT}/models/triton-model-repo | grep paraphrase-multilingual-mpnet-base-v2

    sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    ```

5. **Deploy the Model**:
    - Reload the docker container, specifying that we also need to load paraphrase-multilingual-mpnet-base-v2
    ```bash
    docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 \
     -v $PWD/models:/models nvcr.io/nvidia/tritonserver:24.04-py3 tritonserver \
     --model-repository=/models/triton-model-repo --exit-on-error=false --model-control-mode=explicit  --load-model \
     all-MiniLM-L6-v2 --load-model sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    ```

    - You should see something similar to the following, indicating Triton has successfully loaded the model:
    ```shell
    +----------------------------------+------------------------------------------------------------------------------------------+
    | Option                           | Value                                                                                    |
    +----------------------------------+------------------------------------------------------------------------------------------+
    | server_id                        | triton                                                                                   |
    | server_version                   | 2.35.0                                                                                   |
    | server_extensions                | classification sequence model_repository ... schedule_policy                             |
    | model_repository_path[0]         | /models/triton-model-repo                                                                |
    | model_control_mode               | MODE_EXPLICIT                                                                            |
    | startup_models_0                 | all-MiniLM-L6-v2                                                                         |
    | startup_models_1                 | sentence-transformers/paraphrase-multilingual-mpnet-base-v2                              |
    | strict_model_config              | 0                                                                                        |
    | rate_limit                       | OFF                                                                                      |
    | pinned_memory_pool_byte_size     | 268435456                                                                                |
    | cuda_memory_pool_byte_size{0}    | 67108864                                                                                 |
    | cuda_memory_pool_byte_size{1}    | 67108864                                                                                 |
    | min_supported_compute_capability | 6.0                                                                                      |
    | strict_readiness                 | 1                                                                                        |
    | exit_timeout                     | 30                                                                                       |
    | cache_enabled                    | 0                                                                                        |
    +----------------------------------+------------------------------------------------------------------------------------------+
    ```
6. **Update the Pipeline Call**:
    - Now that the model has been exported and deployed, we can update the pipeline call to use the new model:
    ```bash
    python examples/llm/main.py vdb_upload pipeline --model_name \
     sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    ```

### Running the Langchain Pipeline (Optional)

- Optional guide for running the Langchain pipeline, if applicable.## Developer Docs

- A link to the developer documentation where the README.md is also linked.

> **Note**: This pipeline will, by default, run continuously repeatedly polling the configured RSS sources. To run for a
> fixed number of iterations, add the `--stop_after=N` flag.
