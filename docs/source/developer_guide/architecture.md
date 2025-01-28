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

# Morpheus Architecture

## Overview

The organization of Morpheus can be broken down into four different layers. Working from the top down:

* Orchestration Layer
    * Responsible for coordinating pipelines and facilitating communication.
        * That is, monitoring pipelines, transferring messages between pipelines, starting and stopping pipelines,
          and assigning resources to pipelines.
    * Plays a large role in multi-machine pipelines but works out of the box for single-machine pipelines.
* Pipeline Layer
    * Composed of one or more stages connected by edges.
    * Data moves between stages using buffered channels, and the pipeline will automatically handle backpressure by
      monitoring the amount of data in each edge buffer.
* Stage Layer
    * Main building blocks in Morpheus.
    * Responsible for executing a specific function on incoming data from previous stages in the pipeline.
    * Isolated from each other and can be thought of as black boxes.
    * Composed of one or more nodes, connected by edges.
    * All nodes are guaranteed to operate on the same machine, in the same process space.
* Module Layer
    * Similar to stages, modules can builds and connect multiple nodes.
* Node Layer
    * Smallest building block in Morpheus.
    * Each node operates on the same thread.
    * Composed of one or more operators in the reactive programming style.

## Pipeline Details

Pipelines are a collection of one or more stages that are connected via edges. Data flows from one stage to the next across these edges using buffers. We utilize these buffers to allow stages to process messages at different rates. Once each stage is done processing a message, the pipeline will move it onto the next stage's buffer for processing. This process continues until the message has made it through the entire pipeline.

The main goal of the pipeline is to maximize throughput via parallel execution of the stages. Such that we can utilize hardware optimally and avoid processing individual messages sequentially. Given a multi-stage pipeline consisting of stages 1 and 2. Stage 1 collects its first message from its data source and begins processing it. Once Stage 1 is done with its first message, the resulting output message will be forwarded to Stage 2. At this point, Stage 1 immediately begins processing the next input to the pipeline, while Stage 2 begins work on the output of Stage 1. This allows for multiple messages to be in flight in the pipeline at a time, increasing parallelization.

Utilizing buffers between stages in this way does come at a cost. Increasing the size of the buffers helps improve parallelization by ensuring all stages have some work to do. But this also increases latency since messages can sit in a buffer waiting to be processed. The inverse is also true. Decreasing the buffer sizes improves latency, but can starve some stages of work to do, decreasing parallelization. The pipeline has to walk a fine line of keeping all stages supplied with data with the smallest buffers possible.

## Stage Details

A stage is the fundamental building block in Morpheus and is responsible for performing all of the work in a pipeline. A stage can encapsulate any piece of functionality and is capable of integrating with any service or external library. This freedom allows stages to range from very small Python map functions up to very complex inference stages, which connect to services and work in multiple threads. For example, Morpheus has simple stages for actions like reading and writing to a file and more complex stages like the Triton inference stage, which can send many asynchronous inference requests using shared device memory.

While stages are very flexible, they all comprise three main pieces: identification, type inference, and node creation.

### Identification

The stage identifier is a unique string used in both logging and creating the stage from the CLI.

### Type Inference

To perform work, each stage needs to know what type of data it will be operating on. Morpheus can pass any type of data from stage to stage, the pipeline must ensure compatible types at every edge connection between stages. This process is called stage type inference and is performed during the pipeline build phase.

Stage type inference is necessary because the output type of some stages can depend on the output type of the previous stage. For example, consider a simple pass through stage that passes the input message to the next stage unmodified. If our pass through stage is preceded by a stage generating a string, its output type will be a string. Instead, if it's preceded by a stage generating an integer, its output type will be an integer.

Due to the dynamic nature of the output type of a stage, stages must specify a type inference function that accepts an input type and returns the output type. Starting at the source stages, the pipeline will use this function to determine the output type of the source stages. This result is then be passed to the type inference function of the next stage, until the input and output types of every stage in the pipeline have been determined.

After the build phase, the output types of stages cannot be changed. Returning a different type than specified during the build phase will result in undefined behavior.

### Node Creation

The most important piece of a stage is node creation. The node creation function is responsible for creating the instances of the nodes which will make up a stage. Like a pipeline, stages can be built up of one or more smaller nodes connected by edges.

The difference between stages and nodes is that stages guarantee that the same machine will run all nodes in the same process space. This allows nodes to optimize the information they pass between themselves to ensure maximum performance. For example, two nodes could pass a raw GPU device pointer between them, allowing maximum performance with minimum overhead. Without this guarantee that both nodes are running in the same process space, passing such a low-level piece of information would be unsafe.

## Morpheus Modules

Modules, introduced in the 23.03 release, introduce a new method for defining units of work which are compact, composable, nestable, and fully reusable. Once a module has been defined and registered, it can be used in new and existing pipelines as either a new ModuleStage or loaded directly within the context of an existing stage using `builder.load_module(...)`.
