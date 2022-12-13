<!--
SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Getting Started with Morpheus

There are three ways to get started with Morpheus:
- [Using pre-built Docker containers](#using-pre-built-docker-containers)
- [Building the Morpheus Docker container](#building-the-morpheus-container)
- [Building Morpheus from source](./developer_guide/contributing.md#building-from-source)

The [pre-built Docker containers](#using-pre-built-docker-containers) are the easiest way to get started with the latest release of Morpheus. Instructions on how to download and run these containers, including the necessary data and models, can be found on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/morpheus/collections/morpheus_).

More advanced users, or those who are interested in using the latest pre-release features, will need to [build the Morpheus container](#building-the-morpheus-container) or [build from source](./developer_guide/contributing.md#building-from-source).

## Requirements
- Pascal architecture GPU or better
- NVIDIA driver `450.80.02` or higher
- [Docker](https://docs.docker.com/get-docker/)
- [The NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
- [NVIDIA Triton Inference Server](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver) `22.06` or higher

## Using pre-built Docker containers
### Pulling the Morpheus Image
1. Goto [https://catalog.ngc.nvidia.com/orgs/nvidia/teams/morpheus/containers/morpheus/tags](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/morpheus/containers/morpheus/tags)
1. Choose a version
1. Download the selected version, for example for `22.11`:
```bash
docker pull nvcr.io/nvidia/morpheus/morpheus:22.11-runtime
```

Skip ahead to the [Starting the Morpheus Container](#starting-the-morpheus-container) section.

## Building the Morpheus Container
### Clone the Repository

```bash
MORPHEUS_ROOT=$(pwd)/morpheus
git clone https://github.com/nv-morpheus/Morpheus.git $MORPHEUS_ROOT
cd $MORPHEUS_ROOT
```

### Git LFS

The large model and data files in this repo are stored using [Git Large File Storage (LFS)](https://git-lfs.github.com/). Only those files which are strictly needed to run Morpheus are downloaded by default when the repository is cloned.

The `scripts/fetch_data.py` script can be used to fetch the Morpheus pre-trained models, and other files required for running the training/validation scripts and example pipelines.

Usage of the script is as follows:
```bash
scripts/fetch_data.py fetch <dataset> [<dataset>...]
```

At time of writing the defined datasets are:
* all - Metaset includes all others
* examples - Data needed by scripts in the `examples` subdir
* models - Morpheus models (largest dataset)
* tests - Data used by unittests
* validation - Subset of the models dataset needed by some unittests

To download just the examples and models:
```bash
scripts/fetch_data.py fetch examples models
```

To download the data needed for unittests:
```bash
scripts/fetch_data.py fetch tests validation
```

If `Git LFS` is not installed the before cloning the repository, the `scripts/fetch_data.py` script will fail. If this is the case follow the instructions for installing `Git LFS` from [here](https://git-lfs.github.com/), and then run the following command:
```bash
git lfs install
```

### Build the Container

To assist in building the Morpheus container, several scripts have been provided in the `./docker` directory. To build the "release" container, run the following:

```bash
./docker/build_container_release.sh
```

This will create an image named `nvcr.io/nvidia/morpheus/morpheus:${MORPHEUS_VERSION}-runtime` where `$MORPHEUS_VERSION` is replaced by the output of `git describe --tags --abbrev=0`.

To run the built "release" container, use the following:

```bash
./docker/run_container_release.sh
```

You can specify different Docker images and tags by passing the script the `DOCKER_IMAGE_TAG`, and `DOCKER_IMAGE_TAG` variables respectively. For example, to run version `v22.11.00a` use the following:

```bash
DOCKER_IMAGE_TAG="v22.11.00a-runtime" ./docker/run_container_release.sh
```

## Starting the Morpheus Container
1. Ensure that [The NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) is installed.
1. Start the container downloaded from the previous section:
```bash
docker run --rm -ti --runtime=nvidia --gpus=all --net=host nvcr.io/nvidia/morpheus/morpheus:22.11-runtime bash
```

Note about some of the flags above:
| Flag | Description |
| ---- | ----------- |
| `--runtime=nvidia` | Choose the Nvidia docker runtime, this enables access to the GPU inside the container. This flag isn't needed if the `nvidia` runtime is already set as the default runtime for Docker |
| `--gpus=all` | Specify which GPUs the container has access to.  Alternately a specific GPU could be chosen with `--gpus=<gpu-id>` |
| `--net=host` | Most of the Morpheus pipelines utilize [NVIDIA Triton Inference Server](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver), which will be running in another container. For simplicity we will give the container access to the host system's network, production deployments may opt for an explicit network configuration. |

## Launching Triton Server

Many of the validation tests and example workflows require a Triton server to function.
In a new terminal, from the root of the Morpheus repo, use the following command to launch a Docker container for Triton loading all of the included pre-trained models:

```bash
docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 \
	-v $PWD/models:/models \
	nvcr.io/nvidia/tritonserver:22.08-py3 \
	tritonserver --model-repository=/models/triton-model-repo \
		--exit-on-error=false \
		--log-info=true \
		--strict-readiness=false
```
This will launch Triton using the default network ports (8000 for HTTP, 8001 for GRPC, and 8002 for metrics).

## Running Morpheus

To run Morpheus, users will need to choose from the Morpheus Command Line Interface (CLI) or Python interface. Which interface to use depends on the user's needs, amount of customization, and operating environment. More information on each interface can be found below.

For full example pipelines using both the Python API and command line interface, refer to the [Morpheus Examples](./examples.md).

### Morpheus Python Interface

The Morpheus Python interface allows users to configure their pipelines using a Python script file. This is ideal for users who are working in a Jupyter notebook, and users who need complex initialization logic. Documentation on using both the Morpheus Python & C++ APIs can be found in the [Morpheus Developer Guide](./developer_guide/guides.md).

### Morpheus Command Line Interface (CLI)

The CLI allows users to completely configure a Morpheus pipeline directly from a terminal. This is ideal for users who do not need customized stages and for users configuring a pipeline in Kubernetes. The Morpheus CLI can be invoked using the `morpheus` command and is capable of running linear pipelines as well as additional tools. Instructions for using the CLI can be queried directly in the terminal using `morpheus --help`:

```bash
$ morpheus --help
Usage: morpheus [OPTIONS] COMMAND [ARGS]...

Options:
  --debug / --no-debug            [default: no-debug]
  --log_level [CRITICAL|FATAL|ERROR|WARN|WARNING|INFO|DEBUG]
                                  Specify the logging level to use.  [default:
                                  WARNING]
  --log_config_file FILE          Config file to use to configure logging. Use
                                  only for advanced situations. Can accept
                                  both JSON and ini style configurations
  --plugin TEXT                   Adds a Morpheus CLI plugin. Can either be a
                                  module name or path to a python module
  --version                       Show the version and exit.
  --help                          Show this message and exit.

Commands:
  run    Run one of the available pipelines
  tools  Run a utility tool
```

Each command in the CLI has its own help information. Use `morpheus [command] [...sub-command] --help` to get instructions for each command and sub command. For example:

```bash
$ morpheus run pipeline-nlp inf-triton --help
Configuring Pipeline via CLI
Usage: morpheus run pipeline-nlp inf-triton [OPTIONS]

Options:
  --model_name TEXT               Model name in Triton to send messages to
                                  [required]
  --server_url TEXT               Triton server URL (IP:Port)  [required]
  --force_convert_inputs BOOLEAN  Instructs this stage to forcibly convert all
                                  input types to match what Triton is
                                  expecting. Even if this is set to `False`,
                                  automatic conversion will be done only if
                                  there would be no data loss (i.e. int32 ->
                                  int64).  [default: False]
  --use_shared_memory BOOLEAN     Whether or not to use CUDA Shared IPC Memory
                                  for transferring data to Triton. Using CUDA
                                  IPC reduces network transfer time but
                                  requires that Morpheus and Triton are
                                  located on the same machine  [default:
                                  False]
  --help                          Show this message and exit.  [default:
                                  False]
```

Several examples on using the Morpheus CLI can be found in the [Basic Usage](./examples/basic_usage/README.md) guide along with the other [Morpheus Examples](./examples.md).

#### CLI Stage Configuration

When configuring a pipeline via the CLI, you start with the command `morpheus run pipeline` and then list the stages in order from start to finish. The order that the commands are placed in will be the order that data flows from start to end. The output of each stage will be linked to the input of the next. For example, to build a simple pipeline that reads from Kafka, deserializes messages, serializes them, and then writes to a file, use the following:

```bash
morpheus run pipeline-nlp from-kafka --bootstrap_servers localhost:9092 --input_topic test_pcap deserialize serialize to-file --filename .tmp/temp_out.json
```

The output should be similar to:

```
====Building Pipeline====
Added source: <from-kafka-0; KafkaSourceStage(bootstrap_servers=localhost:9092, input_topic=test_pcap, group_id=custreamz, poll_interval=10millis)>
  └─> morpheus.MessageMeta
Added stage: <deserialize-1; DeserializeStage()>
  └─ morpheus.MessageMeta -> morpheus.MultiMessage
Added stage: <serialize-2; SerializeStage(include=[], exclude=['^ID$', '^_ts_'], output_type=pandas)>
  └─ morpheus.MultiMessage -> pandas.DataFrame
Added stage: <to-file-3; WriteToFileStage(filename=.tmp/temp_out.json, overwrite=False, file_type=auto)>
  └─ pandas.DataFrame -> pandas.DataFrame
====Building Pipeline Complete!====
```

This is important because it shows you the order of the stages and the output type of each one. Since some stages cannot accept all types of inputs, Morpheus will report an error if you have configured your pipeline incorrectly. For example, if we run the same command as above but forget the `serialize` stage, Morpheus should ouput an error similar to:

```bash
$ morpheus run pipeline-nlp from-kafka --input_topic test_pcap deserialize to-file --filename .tmp/temp_out.json --overwrite

====Building Pipeline====
Added source: from-kafka -> <class 'cudf.core.dataframe.DataFrame'>
Added stage: deserialize -> <class 'morpheus.pipeline.messages.MultiMessage'>

Traceback (most recent call last):
  File "morpheus/pipeline/pipeline.py", line 228, in build_and_start
    current_stream_and_type = await s.build(current_stream_and_type)
  File "morpheus/pipeline/pipeline.py", line 108, in build
    raise RuntimeError("The {} stage cannot handle input of {}. Accepted input types: {}".format(
RuntimeError: The to-file stage cannot handle input of <class 'morpheus.pipeline.messages.MultiMessage'>. Accepted input types: (typing.List[str],)
```

This indicates that the `to-file` stage cannot accept the input type of `morpheus.pipeline.messages.MultiMessage`. This is because the `to-file` stage has no idea how to write that class to a file; it only knows how to write strings. To ensure you have a valid pipeline, look at the `Accepted input types: (typing.List[str],)` portion of the message. This indicates you need a stage that converts from the output type of the `deserialize` stage, `morpheus.pipeline.messages.MultiMessage`, to `typing.List[str]`, which is exactly what the `serialize` stage does.

#### Pipeline Stages

A complete list of the pipeline stages will be added in the future. For now, you can query the available stages for each pipeline type via:

```bash
$ morpheus run pipeline-nlp --help
Usage: morpheus run pipeline-nlp [OPTIONS] COMMAND1 [ARGS]... [COMMAND2
                               [ARGS]...]...

<Help Paragraph Omitted>

Commands:
  add-class     Add detected classifications to each message
  add-scores    Add probability scores to each message
  buffer        (Deprecated) Buffer results
  delay         (Deprecated) Delay results for a certain duration
  deserialize   Deserialize source data from JSON.
  dropna        Drop null data entries from a DataFrame
  filter        Filter message by a classification threshold
  from-file     Load messages from a file
  from-kafka    Load messages from a Kafka cluster
  gen-viz       (Deprecated) Write out visualization data frames
  inf-identity  Perform a no-op inference for testing
  inf-pytorch   Perform inference with PyTorch
  inf-triton    Perform inference with Triton
  mlflow-drift  Report model drift statistics to ML Flow
  monitor       Display throughput numbers at a specific point in the pipeline
  preprocess    Convert messages to tokens
  serialize     Serializes messages into a text format
  to-file       Write all messages to a file
  to-kafka      Write all messages to a Kafka cluster
  validate      Validates pipeline output against an expected output
```

And for the FIL pipeline:

```bash
$ morpheus run pipeline-fil --help
Usage: morpheus run pipeline-fil [OPTIONS] COMMAND1 [ARGS]... [COMMAND2
                               [ARGS]...]...

<Help Paragraph Omitted>

Commands:
  add-class     Add detected classifications to each message
  add-scores    Add probability scores to each message
  buffer        (Deprecated) Buffer results
  delay         (Deprecated) Delay results for a certain duration
  deserialize   Deserialize source data from JSON.
  dropna        Drop null data entries from a DataFrame
  filter        Filter message by a classification threshold
  from-file     Load messages from a file
  from-kafka    Load messages from a Kafka cluster
  inf-identity  Perform a no-op inference for testing
  inf-pytorch   Perform inference with PyTorch
  inf-triton    Perform inference with Triton
  mlflow-drift  Report model drift statistics to ML Flow
  monitor       Display throughput numbers at a specific point in the pipeline
  preprocess    Convert messages to tokens
  serialize     Serializes messages into a text format
  to-file       Write all messages to a file
  to-kafka      Write all messages to a Kafka cluster
  validate      Validates pipeline output against an expected output
```

And for the AE pipeline:

```bash
$ morpheus run pipeline-ae --help
Usage: morpheus run pipeline-ae [OPTIONS] COMMAND1 [ARGS]... [COMMAND2
                               [ARGS]...]...

<Help Paragraph Omitted>

Commands:
  add-class        Add detected classifications to each message.
  add-scores       Add probability scores to each message.
  buffer           (Deprecated) Buffer results.
  delay            (Deprecated) Delay results for a certain duration.
  filter           Filter message by a classification threshold.
  from-azure       Source stage is used to load Azure Active Directory messages.
  from-cloudtrail  Load messages from a Cloudtrail directory.
  from-duo         Source stage is used to load Duo Authentication messages.
  inf-pytorch      Perform inference with PyTorch.
  inf-triton       Perform inference with Triton Inference Server.
  monitor          Display throughput numbers at a specific point in the pipeline.
  preprocess       Prepare Autoencoder input DataFrames for inference.
  serialize        Include & exclude columns from messages.
  timeseries       Perform time series anomaly detection and add prediction.
  to-file          Write all messages to a file.
  to-kafka         Write all messages to a Kafka cluster.
  train-ae         Train an Autoencoder model on incoming data.
  trigger          Buffer data until previous stage has completed.
  validate         Validate pipeline output for testing.
```
Note: The available commands for different types of pipelines are not the same. This means that the same stage, when used in different pipelines, may have different options. Please check the CLI help for the most up-to-date information during development.

## Next Steps
* [Morpheus Examples](./examples.md) - Example pipelines using both the Python API and command line interface
* [Morpheus Developer Guide](./developer_guide/guides.md) - Documentation on using the Morpheus Python & C++ APIs
