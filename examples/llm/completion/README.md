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

# Completion Pipeline

## Table of Contents

1. [Background Information](#background-information)
    - [Purpose](#purpose)
        - [LLM Service](#llm-service)
        - [Downstream Tasks](#downstream-tasks)
    - [Pipeline Implementation](#pipeline-implementation)
2. [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
        - [Install Dependencies](#install-dependencies)
        - [Setting up NGC API Key](#setting-up-ngc-api-key)
    - [Running the Morpheus Pipeline](#running-the-morpheus-pipeline)

## Supported Environments
All environments require additional Conda packages which can be installed with either the `conda/environments/all_cuda-125_arch-$(arch).yaml` or `conda/environments/examples_cuda-125_arch-$(arch).yaml` environment files. Refer to the [Install Dependencies](#install-dependencies) section for more information.
| Environment | Supported | Notes |
|-------------|-----------|-------|
| Conda | ✔ | |
| Morpheus Docker Container | ✔ |  |
| Morpheus Release Container | ✔ |  |
| Dev Container | ✔ |  |


## Background Information

### Purpose

The primary goal of this example is to showcase the creation of a pipeline that integrates an LLM service with Morpheus. Although this example features a single implementation, the pipeline and its components are versatile and can be adapted to various scenarios with unique requirements. The following highlights different customization points within the pipeline and the specific choices made for this example:

#### LLM Service

- The pipeline is designed to support any LLM service that adheres to our LLMService interface. Compatible services include OpenAI, NeMo, or even local execution using llama-cpp-python. In this demonstration, we focus on utilizing NeMo as the LLM service, highlighting the advantages it offers over other LLM services and the seamless integration with the NeMo ecosystem. Furthermore, the pipeline can accommodate more complex configurations using NeMo + Inform without necessitating changes to the core pipeline.

#### Downstream Tasks

- Post LLM execution, the model's output can be leveraged for various tasks, including model training, analysis, or simulating an attack. In this particular example, we have simplified the implementation and focused solely on the LLMEngine.

### Pipeline Implementation

This example Morpheus pipeline is built using the following components:

- **InMemorySourceStage**: Manages LLM queries in a DataFrame.
- **DeserializationStage**: Converts MessageMeta objects into ControlMessages required by the LLMEngine.
- **LLMEngineStage**: Encompasses the core LLMEngine functionality.
    - An `ExtracterNode` extracts the questions from the DataFrame.
    - A `PromptTemplateNode` converts data and a template into the final inputs for the LLM.
    - The LLM executes using an `LLMGenerateNode` to run the LLM queries.
    - Finally, the responses are incorporated back into the ControlMessage using a `SimpleTaskHandler`.
- **InMemorySinkStage**: Store the results.

## Getting Started

### Prerequisites

Before running the pipeline, ensure that the `NGC_API_KEY` environment variable is set.

#### Install Dependencies

Install the required dependencies.

```bash
conda env update --solver=libmamba \
  -n ${CONDA_DEFAULT_ENV} \
  --file ./conda/environments/examples_cuda-125_arch-$(arch).yaml
```


#### Setting up NGC API Key

For this example, we utilize the NeMo Service within NGC. To gain access, an NGC API key is required. Follow the
instructions outlined [here](https://docs.nvidia.com/ngc/gpu-cloud/ngc-user-guide/index.html#generating-api-key) to
generate your NGC API key.

Configure the following environment variables, with NGC_ORG_ID being optional:

```bash
export NGC_API_KEY=<YOUR_API_KEY>
export NGC_ORG_ID=<YOUR_NGC_ORG_ID>
```

### Running the Morpheus Pipeline

The top level entrypoint to each of the LLM example pipelines is `examples/llm/main.py`. This script accepts a set
of Options and a Pipeline to run. Baseline options are below, and for the purposes of this document we'll assume a
pipeline option of `completion`:

### Run example:

```bash
python examples/llm/main.py completion [OPTIONS] COMMAND [ARGS]...
```

#### Commands:

- `pipeline`

##### Options:
- `--use_cpu_only`
    - **Description**: Run in CPU only mode
    - **Default**: `False`

- `--num_threads INTEGER RANGE`
    - **Description**: Number of internal pipeline threads to use.
    - **Default**: `12`

- `--pipeline_batch_size INTEGER RANGE`
    - **Description**: Internal batch size for the pipeline. Can be much larger than the model batch size.
      Also used for Kafka consumers.
    - **Default**: `1024`

- `--model_max_batch_size INTEGER RANGE`
    - **Description**: Max batch size to use for the model.
    - **Default**: `64`

- `--repeat_count INTEGER RANGE`
    - **Description**: Number of times to repeat the input query. Useful for testing performance.
    - **Default**: `64`

- `--llm_service [NemoLLM|OpenAI]`
    - **Description**: LLM service to issue requests to.
    - **Default**: `NemoLLM`

- `--help`
    - **Description**: Show the help message with options and commands details.

### Running Morpheus Pipeline with OpenAI LLM service

```bash
python examples/llm/main.py completion pipeline --llm_service OpenAI
```
