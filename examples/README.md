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

# Examples
* [Anomalous Behavior Profiling with Forest Inference Library (FIL) Example](./abp_nvsmi_detection/README.md)
* [ABP Detection Example Using Morpheus](./abp_pcap_detection/README.md)
* [Digital Fingerprinting (DFP)](./digital_fingerprinting/README.md)
* [GNN Fraud Detection Pipeline](./gnn_fraud_detection_pipeline/README.md)
* [Example cyBERT Morpheus Pipeline for Apache Log Parsing](./log_parsing/README.md)
* [Sensitive Information Detection with Natural Language Processing (NLP) Example](./nlp_si_detection/README.md)
* [Example Ransomware Detection Morpheus Pipeline for AppShield Data](./ransomware_detection/README.md)
* [Root Cause Analysis Acceleration & Predictive Maintenance Example](./root_cause_analysis/README.md)
* [SID Visualization Example](./sid_visualization/README.md)
* Large Language Models (LLMs)
  * [Agents](./llm/agents/README.md)
  * [Completion](./llm/completion/README.md)
  * [VDB Upload](./llm/vdb_upload/README.md)
  * [Retrieval Augmented Generation (RAG)](./llm/rag/README.md)

## Environments
Morpheus supports multiple environments, each environment is intended to support a given use-case. Each example documents which environments it is able to run in. With the exception of the Morpheus Release Container, the examples require fetching both the `datasets` and `examples` dataset via the `fetch_data.sh` script:
```bash
./scripts/fetch_data.py fetch examples datasets
```

The following are the supported environments:
| Environment | Description |
|-------------|-------------|
| [Conda](../docs/source/developer_guide/contributing.md#build-in-a-conda-environment) | Morpheus is built from source by the end user, and dependencies are installed via the Conda package manager. |
| [Morpheus Docker Container](../docs/source/developer_guide/contributing.md#build-in-docker-container) | A Docker container that is built from source by the end user, Morpheus is then built from source from within the container. |
| [Morpheus Release Container](../docs/source/getting_started.md#building-the-morpheus-container) | Pre-built Docker container that is built from source by the Morpheus team, and is available for download from the [NGC container registry](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/morpheus/containers/morpheus/tags), or can be built locally from source. |
| [Dev Container](../.devcontainer/README.md) | A [Dev Container](https://containers.dev/) that is built from source by the end user, Morpheus is then built from source from within the container. |
