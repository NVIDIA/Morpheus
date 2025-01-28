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

# Glossary of Common Morpheus Terms

<!-- Please keep these sorted alphabetically -->
## MLflow Triton Plugin
Docker container published on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/morpheus/containers/mlflow-triton-plugin), allowing the deployment of models in [MLflow](https://mlflow.org/) to [Triton Inference Server](#triton-inference-server). This is also available as a [Helm Chart](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/morpheus/helm-charts/morpheus-mlflow).

## module
A Morpheus module is a type of work unit that can be utilized in the Morpheus stage and can be registered to a MRC segment module registry. Modules are beneficial when there is a possibility for the work-unit to be reused.

## Morpheus AI Engine
A Helm Chart for deploying the infrastructure of Morpheus. It includes the [Triton Inference Server](#triton-inference-server), Kafka, and Zookeeper. Refer to [https://catalog.ngc.nvidia.com/orgs/nvidia/teams/morpheus/helm-charts/morpheus-ai-engine](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/morpheus/helm-charts/morpheus-ai-engine).

## Morpheus SDK CLI
A Helm Chart that deploys the Morpheus container. Refer to [https://catalog.ngc.nvidia.com/orgs/nvidia/teams/morpheus/helm-charts/morpheus-sdk-client](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/morpheus/helm-charts/morpheus-sdk-client)

## `morpheus-sdk-client`
Another name for the [Morpheus SDK CLI](#morpheus-sdk-cli) Helm Chart.

## MRC
[Morpheus Runtime Core (MRC)](https://github.com/nv-morpheus/MRC). Pipelines in MRC are low level representations of Morpheus [pipelines](#pipeline).

## NGC
[NVIDIA GPU Cloud](https://catalog.ngc.nvidia.com/) is the official location for Morpheus and many other NVIDIA Docker containers.

## node
An individual node in the MRC pipeline. In Morpheus, MRC nodes are constructed by [stages](#stage).

## operator
Refers to small-reusable MRC nodes contained in the `mrc.core.operators` Python module which perform common tasks such as:
* `filter`
* `flatten`
* `map`
* `on_completed`
* `pairwise`
* `to_list`

## pipeline
Represents all work to be performed end-to-end in Morpheus. A Morpheus pipeline consists of one or more [segments](#segment), and each segment consists of one or more [stages](#stage). At build time, a Morpheus pipeline is transformed into a [MRC](#mrc) pipeline which is then executed.

## RxCpp
MRC is built on top of [RxCpp](https://github.com/ReactiveX/RxCpp) which is an open source C++ implementation of the [ReactiveX API](https://reactivex.io/). In general, Morpheus users are only exposed to this when they wish to write a [stage](#stage) in C++.

## segment
A subgraph of a [pipeline](#pipeline). Segments allow for both logical grouping, and distribution across multiple processes and execution hosts.

## stage
Fundamental building block in Morpheus representing a unit of work. Stages may consist of a single MRC node, a small collection of nodes, or an entire MRC subgraph. A stage can encapsulate any piece of functionality and is capable of integrating with any service or external library. Refer to [Simple Python Stage](../developer_guide/guides/1_simple_python_stage.md).

## Triton Inference Server
Triton Inference Server, part of the NVIDIA AI platform, streamlines and standardizes AI inference by enabling teams to deploy, run, and scale trained AI models from any framework on any GPU- or CPU-based infrastructure. Most Morpheus pipelines utilize Triton for inferencing via the `TritonInferenceStage`. Refer to [https://developer.nvidia.com/nvidia-triton-inference-server](https://developer.nvidia.com/nvidia-triton-inference-server)
