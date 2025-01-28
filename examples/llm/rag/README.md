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

# Retrieval Augmented Generation (RAG) Pipeline

## Supported Environments
All environments require additional Conda packages which can be installed with either the `conda/environments/all_cuda-125_arch-$(arch).yaml` or `conda/environments/examples_cuda-125_arch-$(arch).yaml` environment files. This example also requires the [VDB upload](../vdb_upload/README.md) pipeline to have been run previously.
| Environment | Supported | Notes |
|-------------|-----------|-------|
| Conda | ✔ | |
| Morpheus Docker Container | ✔ | Requires launching Milvus on the host |
| Morpheus Release Container | ✔ | Requires launching Milvus on the host |
| Dev Container | ✘ |  |

### Supported Architectures
| Architecture | Supported | Issue |
|--------------|-----------|-------|
| x86_64 | ✔ | |
| aarch64 | ✘ | [#2122](https://github.com/nv-morpheus/Morpheus/issues/2122) |

## Table of Contents

## Background Information

### Purpose

The purpose of this example is to illustrate how a user could build a Retrieval Augmented Generation pipeline
integrating informational feeds and an LLM service into a Morpheus pipeline. This example builds on the previous
[Completion Pipeline](../completion/README.md) example, by adding the ability to augment LLM queries with context
information from a knowledge base. Appending this context helps improve the responses from the LLM by providing
additional background contextual and factual information which the LLM can pull from for its response.

### Source Documents

- In order for this pipeline to function correctly, a Vector Database must already have been populated with information
  that can be retrieved.
- An example of populating a database is illustrated in [VDB upload](../vdb_upload/README.md)
- This example assumes that pipeline has already been run to completion.

### Vector Database Service

- Any vector database can be used to store the resulting embedding and corresponding metadata.
- It would be trivial to update the example to use Chroma or FAISS if needed.
- For this example, we will be using Milvus since it is the default VDB used in
  the [VDB upload](../vdb_upload/README.md) pipeline.

### Implementation and Design Decisions

### Implementation Details

[Original GitHub issue](https://github.com/nv-morpheus/Morpheus/issues/1306)

In order to cater to the unique requirements of the Retrieval Augmented Generation (RAG) mechanism, the following steps
were incorporated:

- **Embedding Retrieval:** Before the LLM can make a completion, relevant context is retrieved from the Vector Database.
  This context is in the form of embeddings that represent pieces of information closely related to the query.
- **Context Augmentation:** The retrieved context is then appended to the user's query, enriching it with the necessary
  background to assist the LLM in generating a more informed completion.
- **LLM Query Execution:** The augmented query is then sent to the LLM, which generates a response based on the
  combination of the original query and the appended context.

### Rationale Behind Design Decisions

- **Using Milvus as VDB:** Milvus offers scalable and efficient vector search capabilities, making it a natural choice
  for embedding retrieval in real-time.
- **Flexible LLM integration:** The LLM is integrated into the pipeline as a standalone component, which allows for
  easy swapping of models and ensures that the pipeline can be easily extended to support multiple LLMs.

### Standalone Morpheus Pipeline

The standalone Morpheus pipeline is built using the following components:

- An `InMemorySourceStage` to hold the LLM queries in a DataFrame.
    - We supply a fixed set of questions in a `source_df` which are then processed by the `LLMEngineStage`
- A `DeserializationStage` to convert `MessageMeta` objects into `ControlMessage` objects as needed by the `LLMEngine`.
    - New functionality was added to the `DeserializeStage` to support `ControlMessage`s and add a default task to each message.
- An `LLMEngineStage` then wraps the core `LLMEngine` functionality.
    - An `ExtracterNode` pulls the questions out of the DataFrame.
    - A `RAGNode` performs the retrieval and adds the context to the query using the supplied template and executes the LLM.
    - Finally, the responses are put back into the `ControlMessage` using a `SimpleTaskHandler`.
- The pipeline concludes with an `InMemorySink` stage to store the results.

> **Note:** For this to function correctly, the VDB upload pipeline must have been run previously.


## Prerequisites

Before running the pipeline, we need obtain service API keys for the following services:

### Obtain an OpenAI API or NGC API Key

#### NGC

- Follow the instructions [here](https://docs.nvidia.com/ngc/gpu-cloud/ngc-user-guide/index.html#generating-personal-api-key)
- We'll refer to your NGC API key as `${NGC_API_KEY}` for the rest of this document.

#### OpenAI

- Follow the instructions [here](https://platform.openai.com/docs/quickstart?context=python) to obtain an OpenAI
  API key.
- We'll refer to your OpenAI API key as `${OPENAI_API_KEY}` for the rest of this document.

Before running the pipeline, we need to ensure that the following services are running:

### Milvus Service

- Follow the instructions [here](https://milvus.io/docs/install_standalone-docker.md) to install and run a Milvus
  service.


### Running the Morpheus Pipeline

The top level entrypoint to each of the LLM example pipelines is `examples/llm/main.py`. This script accepts a set
of Options and a Pipeline to run. Baseline options are below, and for the purposes of this document we'll assume a
pipeline option of `rag`:

### Run example (Standalone Pipeline):

**Using NGC NeMo LLMs**

```bash
export NGC_API_KEY=[YOUR_KEY_HERE]
python examples/llm/main.py rag pipeline
```

**Using OpenAI LLM models**

```bash
export OPENAI_API_KEY=[YOUR_KEY_HERE]
python examples/llm/main.py rag pipeline --llm_service=OpenAI --model_name=gpt-3.5-turbo
```
