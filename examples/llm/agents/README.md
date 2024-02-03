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

# Morpheus LLM Agents Pipeline

## Table of Contents

1. [Background Information](#background-information)
    - [Purpose](#purpose)
    - [LLM Service](#llm-service)
    - [Agent type](#agent-type)
    - [Agent tools](#agent-tools)
    - [LLM Library](#llm-library)
2. [Pipeline Implementation](#pipeline-implementation)
3. [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
        - [Set Environment Variables](#set-environment-variables)
        - [Install Dependencies](#install-dependencies)
    - [Running the Morpheus Pipeline](#running-the-morpheus-pipeline)
        - [Run example (Simple Pipeline)](#run-example-simple-pipeline)
        - [Run example (Kafka Pipeline)](#run-example-kafka-pipeline)

# Background Information

### Purpose
The Morpheus LLM Agents pipeline is designed to seamlessly integrate Large Language Model (LLM) agents into the Morpheus framework. This implementation focuses on efficiently executing multiple LLM queries using the ReAct agent type, which is tailored for versatile task handling. The use of the Langchain library streamlines the process, minimizing the need for additional system migration.

Within the Morpheus LLM Agents context, these agents act as intermediaries, facilitating communication between users and the LLM service. Their primary role is to execute tools and manage multiple LLM queries, enhancing the LLM's capabilities in solving complex tasks. Agents utilize various tools, such as internet searches, VDB retrievers, calculators, and more, to assist in resolving inquiries, enabling seamless execution of tasks and efficient handling of diverse queries.

### LLM Service
This pipeline supports various LLM services compatible with our LLMService interface, including OpenAI, NeMo, or local execution using llama-cpp-python. In this example, we'll focus on using OpenAI, chosen for its compatibility with the ReAct agent architecture.

### Agent type
The pipeline supports different agent types, each influencing the pattern for interacting with the LLM. For this example, we'll use the ReAct agent type—a popular and reliable choice.

### Agent tools
Depending on the problem at hand, various tools can be provided to LLM agents, such as internet searches, VDB retrievers, calculators, Wikipedia, etc. In this example, we'll use the internet search tool and an llm-math tool, allowing the LLM agent to perform Google searches and solve math equations.

### LLM Library
The pipeline utilizes the Langchain, Haystack library to run LLM agents, enabling their execution directly within a Morpheus pipeline. This approach reduces the overhead of migrating existing systems to Morpheus and eliminates the need to replicate work done by popular LLM libraries like llama-index and Haystack.

## Pipeline Implementation
- **InMemorySourceStage**: Manages LLM queries in a DataFrame.
- **KafkaSourceStage**: Consumes LLM queries from the Kafka topic.
- **DeserializationStage**: Converts MessageMeta objects into ControlMessages required by the LLMEngine.
- **LLMEngineStage**: Encompasses the core LLMEngine functionality.
    - An `ExtracterNode` extracts the questions from the DataFrame.
    - A `LangChainAgentNode` runs the Langchain agent executor for all provided input. This node will utilize the agents
     run interface to run the agents asynchronously.
    - Finally, the responses are incorporated back into the ControlMessage using a `SimpleTaskHandler`.
- **InMemorySinkStage**: Store the results.

## Getting Started

### Prerequisites

#### Set Environment Variables

Before running the project, ensure that you set the required environment variables. Follow the steps below to obtain and set the API keys for OpenAI and SerpApi.

**OpenAI API Key**

Visit [OpenAI](https://openai.com/) and create an account. Navigate to your account settings to obtain your OpenAI API
key. Copy the key and set it as an environment variable using the following command:

```bash
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
```

**SerpApi API Key**

Go to [SerpApi](https://serpapi.com/users/sign_up) to register and create an account. Once registered, obtain your
SerpApi API key. Set the API key as an environment variable using the following command:

```bash
export SERPAPI_API_KEY="<YOUR_SERPAPI_API_KEY>"
```

**Serper Dev API Key**

Go to [SerperDev](https://serper.dev/login) to register and create an account. Once registered, obtain your
Serper Dev API key. Set the API key as an environment variable using the following command:

```bash
export SERPERDEV_API_KEY="<SERPER_API_KEY>"
```

Note: This is required when using the Haystack LLM orchestration framework in the pipeline.

#### Install Dependencies

Install the required dependencies.

```bash
export CUDA_VER=11.8
mamba install -n base -c conda-forge conda-merge
conda run -n base --live-stream conda-merge docker/conda/environments/cuda${CUDA_VER}_dev.yml \
  docker/conda/environments/cuda${CUDA_VER}_examples.yml > .tmp/merged.yml \
  && mamba env update -n ${CONDA_DEFAULT_ENV} --file .tmp/merged.yml
```


### Running the Morpheus Pipeline

The top level entrypoint to each of the LLM example pipelines is `examples/llm/main.py`. This script accepts a set
of Options and a Pipeline to run. Baseline options are below, and for the purposes of this document we'll assume a
pipeline option of `agents`:

### Run example (Simple Pipeline):

This example demonstrates the basic implementation of Morpheus pipeline, showcasing the process of executing LLM queries and managing the generated responses. It uses different stages such as InMemorySourceStage, DeserializationStage, ExtracterNode, LangChainAgentNode, SimpleTaskHandler, and InMemorySinkStage within the pipeline to handle various aspects of query processing and response management.

- Utilizes stages such as InMemorySourceStage and DeserializationStage for consuming and batching LLM queries.
- Incorporates an ExtracterNode for extracting questions and a LangChainAgentNode for executing the Langchain agent executor.
- SimpleTaskHandler to manage the responses generated by the LLMs.
- Stores and manages the results within the pipeline using an InMemorySinkStage.


```bash
python examples/llm/main.py agents simple [OPTIONS]
```

### Options:
- `--num_threads INTEGER RANGE`
    - **Description**: Number of internal pipeline threads to use.
    - **Default**: `12`

- `--pipeline_batch_size INTEGER RANGE`
    - **Description**: Internal batch size for the pipeline. Can be much larger than the model batch size. Also
    used for Kafka consumers.
    - **Default**: `1024`

- `--model_max_batch_size INTEGER RANGE`
    - **Description**: Max batch size to use for the model.
    - **Default**: `64`

- `--model_name TEXT`
    - **Description**: The name of the model to use in OpenAI.
    - **Default**: `gpt-3.5-turbo-instruct`

- `--repeat_count INTEGER RANGE`
    - **Description**: Number of times to repeat the input query. Useful for testing performance.
    - **Default**: `1`

- `--llm_orch TEXT`
    - **Chioce**: `[haystack|langchain|llama_index]`
    - **Description**: The name of the model to use in OpenAI.
    - **Default**: `langchain`

- `--help`
    - **Description**: Show the help message with options and commands details.


### Run example (Kafka Pipeline):

The Kafka Example in the Morpheus LLM Agents demonstrates an streaming implementation, utilizing Kafka messages to
facilitate the near real-time processing of LLM queries. This example is similar to the Simple example but makes use of
a KafkaSourceStage to stream and retrieve messages from the Kafka topic

First, to run the Kafka example, you need to create a Kafka cluster that enables the persistent pipeline to accept queries for the LLM agents. You can create the Kafka cluster using the following guide: [Quick Launch Kafka Cluster Guide](https://github.com/nv-morpheus/Morpheus/blob/branch-23.11/docs/source/developer_guide/contributing.md#quick-launch-kafka-cluster)

Once the Kafka cluster is running, create Kafka topic to produce input to the pipeline.

```bash
# Set the bootstrap server variable
export BOOTSTRAP_SERVER=$(broker-list.sh)

# Create the input and output topics
kafka-topics.sh --bootstrap-server ${BOOTSTRAP_SERVER} --create --topic input

# Update the partitions
kafka-topics.sh --bootstrap-server ${BOOTSTRAP_SERVER} --alter --topic input --partitions 3
```

Now Kafka example can be run using the following command with the below listed options:

```bash
python examples/llm/main.py agents kafka [OPTIONS]
```

### Options:
- `--num_threads INTEGER RANGE`
    - **Description**: Number of internal pipeline threads to use.
    - **Default**: `12`

- `--pipeline_batch_size INTEGER RANGE`
    - **Description**: Internal batch size for the pipeline. Can be much larger than the model batch size. Also
    used for Kafka consumers.
    - **Default**: `1024`

- `--model_max_batch_size INTEGER RANGE`
    - **Description**: Max batch size to use for the model.
    - **Default**: `64`

- `--model_name TEXT`
    - **Description**: The name of the model to use in OpenAI.
    - **Default**: `gpt-3.5-turbo-instruct`

- `--llm_orch TEXT`
    - **Chioce**: `[haystack|langchain]`
    - **Description**: The name of the model to use in OpenAI.
    - **Default**: `langchain`

- `--help`
    - **Description**: Show the help message with options and commands details.

After the pipeline is running, we need to send messages to the pipeline using the Kafka topic. In a separate terminal, run the following command:

```bash
kafka-console-producer.sh --bootstrap-server ${BOOTSTRAP_SERVER} --topic input
```

This will open up a prompt allowing any JSON to be pasted into the terminal. The JSON should be formatted as follows:

```json
{"question": "<Your question here>"}
```

For example:
```json
{"question": "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"}
{"question": "What is the height of the tallest mountain in feet divided by 2.23? Do not round your answer"}
{"question": "Who is the current leader of Japan? What is the largest prime number that is smaller that their age? Just say the number."}
```
