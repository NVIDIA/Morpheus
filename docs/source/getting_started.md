<!--
SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

The [pre-built Docker containers](#using-pre-built-docker-containers) are the easiest way to get started with the latest release of Morpheus. Released versions of Morpheus containers can be found on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/morpheus/collections/morpheus_).

More advanced users, or those who are interested in using the latest pre-release features, will need to [build the Morpheus container](#building-the-morpheus-container) or [build from source](./developer_guide/contributing.md#building-from-source).

## Requirements
- Volta architecture GPU or better
- [CUDA 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive)
- [Docker](https://docs.docker.com/get-docker/)
- [The NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
- [NVIDIA Triton Inference Server](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver) `24.04` or higher

> **Note about Docker:**
>
> The Morpheus documentation and examples assume that the [Manage Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user) post install step has been performed allowing Docker commands to be executed by a non-root user. This is not strictly necessary so long as the current user has `sudo` privileges to execute Docker commands.

## Using pre-built Docker containers
### Pull the Morpheus Image
1. Go to [https://catalog.ngc.nvidia.com/orgs/nvidia/teams/morpheus/containers/morpheus/tags](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/morpheus/containers/morpheus/tags)
1. Choose a version
1. Download the selected version, for example for `24.06`:
```bash
docker pull nvcr.io/nvidia/morpheus/morpheus:24.06-runtime
```

> **Note about Morpheus versions:**
>
> Morpheus uses Calendar Versioning ([CalVer](https://calver.org/)). For each Morpheus release there will be an image tagged in the form of `YY.MM-runtime` this tag will always refer to the latest point release for that version. In addition to this there will also be at least one point release version tagged in the form of `vYY.MM.00-runtime` this will be the initial point release for that version (ex. `v24.06.00-runtime`). In the event of a major bug, we may release additional point releases (ex. `v24.06.01-runtime`, `v24.06.02-runtime` etc...), and the `YY.MM-runtime` tag will be updated to reference that point release.
>
> Users who want to ensure they are running with the latest bug fixes should use a release image tag (`YY.MM-runtime`). Users who need to deploy a specific version into production should use a point release image tag (`vYY.MM.00-runtime`).

### Starting the Morpheus Container
1. Ensure that [The NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) is installed.
1. Start the container downloaded from the previous section:
```bash
docker run --rm -ti --runtime=nvidia --gpus=all --net=host -v /var/run/docker.sock:/var/run/docker.sock nvcr.io/nvidia/morpheus/morpheus:24.06-runtime bash
```

Note about some of the flags above:
| Flag | Description |
| ---- | ----------- |
| `--runtime=nvidia` | Choose the NVIDIA docker runtime, this enables access to the GPU inside the container. This flag isn't needed if the `nvidia` runtime is already set as the default runtime for Docker. |
| `--gpus=all` | Specify which GPUs the container has access to.  Alternately, a specific GPU could be chosen with `--gpus=<gpu-id>` |
| `--net=host` | Most of the Morpheus pipelines utilize [NVIDIA Triton Inference Server](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver), which will be running in another container. For simplicity we will give the container access to the host system's network, production deployments may opt for an explicit network configuration. |
| `-v /var/run/docker.sock:/var/run/docker.sock` | Enables access to the Docker socket file from within the running container, this allows launching other Docker containers from within the Morpheus container. This flag is required for launching Triton with access to the included Morpheus models, users with their own models can omit this. |

Once launched, users wishing to launch Triton using the included Morpheus models will need to install the Docker tools in the Morpheus container by running:
```bash
./external/utilities/docker/install_docker.sh
```

Skip ahead to the [Launching Triton Server](#launching-triton-server) section.

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
* docs - Graphics needed for documentation
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

If `Git LFS` is not installed before cloning the repository, the `scripts/fetch_data.py` script will fail. If this is the case, follow the instructions for installing `Git LFS` from [here](https://git-lfs.github.com/), and then run the following command:
```bash
git lfs install
```

### Build the Container

To assist in building the Morpheus container, several scripts have been provided in the `./docker` directory. To build the "release" container, run the following:

```bash
./docker/build_container_release.sh
```

By default this will create an image named `nvcr.io/nvidia/morpheus/morpheus:${MORPHEUS_VERSION}-runtime` where `$MORPHEUS_VERSION` is replaced by the output of `git describe --tags --abbrev=0`. You can specify a different Docker image name and tag by passing the script the `DOCKER_IMAGE_NAME`, and `DOCKER_IMAGE_TAG` environment variables respectively.

To run the built "release" container, use the following:

```bash
./docker/run_container_release.sh
```

The `./docker/run_container_release.sh` script accepts the same `DOCKER_IMAGE_NAME`, and `DOCKER_IMAGE_TAG` environment variables that the `./docker/build_container_release.sh` script does. For example, to run version `v24.06.00` use the following:

```bash
DOCKER_IMAGE_TAG="v24.06.00-runtime" ./docker/run_container_release.sh
```

## Launching Triton Server

Many of the validation tests and example workflows require a Triton server to function. In a new terminal, from the root of the Morpheus repo, use the following command to launch a Docker container for Triton loading all of the included pre-trained models:

```bash
docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 \
  -v $PWD/models:/models \
  nvcr.io/nvidia/tritonserver:24.04-py3 \
  tritonserver --model-repository=/models/triton-model-repo \
    --exit-on-error=false \
    --log-info=true \
    --strict-readiness=false \
    --disable-auto-complete-config
```

This will launch Triton using the default network ports (8000 for HTTP, 8001 for GRPC, and 8002 for metrics), loading all of the examples models in the Morpheus repo.

Note: The above command is useful for testing out Morpheus, however it does load several models into GPU memory, which at time of writing consumes roughly 2GB of GPU memory. Production users should consider only loading the specific model(s) they plan on using with the `--model-control-mode=explicit` and `--load-model` flags. For example to launch Triton only loading the `abp-nvsmi-xgb` model:
```bash
docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 \
  -v $PWD/models:/models \
  nvcr.io/nvidia/tritonserver:24.04-py3 \
  tritonserver --model-repository=/models/triton-model-repo \
    --exit-on-error=false \
    --log-info=true \
    --strict-readiness=false \
    --disable-auto-complete-config \
    --model-control-mode=explicit \
    --load-model abp-nvsmi-xgb
```

## Running Morpheus

To run Morpheus, users will need to choose from the Morpheus Command Line Interface (CLI) or Python interface. Which interface to use depends on the user's needs, amount of customization, and operating environment. More information on each interface can be found below.

For full example pipelines using both the Python API and command line interface, refer to the [Morpheus Examples](./examples.md).

### Morpheus Python Interface

The Morpheus Python interface allows users to configure their pipelines using a Python script file. This is ideal for users who are working in a Jupyter Notebook, and users who need complex initialization logic. Documentation on using both the Morpheus Python & C++ APIs can be found in the [Morpheus Developer Guide](./developer_guide/guides.md).

### Morpheus Command Line Interface (CLI)

The CLI allows users to completely configure a Morpheus pipeline directly from a terminal. This is ideal for users configuring a pipeline in Kubernetes. The Morpheus CLI can be invoked using the `morpheus` command and is capable of running linear pipelines as well as additional tools. Instructions for using the CLI can be queried directly in the terminal using `morpheus --help`:

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

Several examples on using the Morpheus CLI can be found in the [Basic Usage](./basics/overview.rst) guide along with the other [Morpheus Examples](./examples.md).

#### CLI Stage Configuration

When configuring a pipeline via the CLI, you start with the command `morpheus run pipeline` and then list the stages in order from start to finish. The order that the commands are placed in will be the order that data flows from start to end. The output of each stage will be linked to the input of the next. For example, to build a simple pipeline that reads from Kafka, deserializes messages, serializes them, and then writes to a file, use the following:

```bash
morpheus --log_level=INFO run pipeline-nlp from-kafka --bootstrap_servers localhost:9092 --input_topic test_pcap deserialize serialize to-file --filename .tmp/temp_out.json --overwrite
```

The output should contain lines similar to:
```
====Building Segment: linear_segment_0====
Added source: <from-kafka-0; KafkaSourceStage(bootstrap_servers=localhost:9092, input_topic=('test_pcap',), group_id=morpheus, client_id=None, poll_interval=10millis, disable_commit=False, disable_pre_filtering=False, auto_offset_reset=AutoOffsetReset.LATEST, stop_after=0, async_commits=True)>
  └─> morpheus.MessageMeta
Added stage: <deserialize-1; DeserializeStage(ensure_sliceable_index=True)>
  └─ morpheus.MessageMeta -> morpheus.MultiMessage
Added stage: <serialize-2; SerializeStage(include=(), exclude=('^ID$', '^_ts_'), fixed_columns=True)>
  └─ morpheus.MultiMessage -> morpheus.MessageMeta
Added stage: <to-file-3; WriteToFileStage(filename=.tmp/temp_out.json, overwrite=True, file_type=FileTypes.Auto, include_index_col=True, flush=False)>
  └─ morpheus.MessageMeta -> morpheus.MessageMeta
====Building Segment Complete!====
====Pipeline Started====
```

This is important because, when the log level is set to `INFO` and above, it shows you the order of the stages and the output type of each one. Since some stages cannot accept all types of inputs, Morpheus will report an error if you have configured your pipeline incorrectly. For example, if we run the same command as above but forget the `serialize` stage, Morpheus should output an error similar to:

```bash
$ morpheus run pipeline-nlp from-kafka --bootstrap_servers localhost:9092 --input_topic test_pcap deserialize to-file --filename .tmp/temp_out.json --overwrite
Configuring Pipeline via CLI
Starting pipeline via CLI... Ctrl+C to Quit
E20221214 14:53:17.425515 452045 controller.cpp:62] exception caught while performing update - this is fatal - issuing kill
E20221214 14:53:17.425714 452045 context.cpp:125] rank: 0; size: 1; tid: 140065439217216; fid: 0x7f6144041000: set_exception issued; issuing kill to current runnable. Exception msg: RuntimeError: The to-file stage cannot handle input of <class 'morpheus.messages.multi_message.MultiMessage'>. Accepted input types: (<class 'morpheus.messages.message_meta.MessageMeta'>,)
```

This indicates that the `to-file` stage cannot accept the input type of `morpheus.messages.multi_message.MultiMessage`. This is because the `to-file` stage has no idea how to write that class to a file; it only knows how to write messages of type `morpheus.messages.message_meta.MessageMeta`. To ensure you have a valid pipeline, examine at the `Accepted input types: (<class 'morpheus.messages.message_meta.MessageMeta'>,)` portion of the error message. This indicates you need a stage that converts from the output type of the `deserialize` stage, `morpheus.messages.multi_message.MultiMessage`, to `morpheus.messages.message_meta.MessageMeta`, which is exactly what the `serialize` stage does.

#### Pipeline Stages

The CLI allows for an easy way to query the available stages for each pipeline type. Refer to [Morpheus Stages](./stages/morpheus_stages.md) for more extensive documentation of the stages included in Morpheus.

> **Note**: While most stages are available from the CLI, a small subset of stages, and configuration options for stages are unavailable from the CLI and can only be used via the Python API.

```
$ morpheus run pipeline-nlp --help
Usage: morpheus run pipeline-nlp [OPTIONS] COMMAND1 [ARGS]... [COMMAND2 [ARGS]...]...

<Help Paragraph Omitted>

Commands:
  add-class     Add detected classifications to each message.
  add-scores    Add probability scores to each message.
  buffer        (Deprecated) Buffer results.
  delay         (Deprecated) Delay results for a certain duration.
  deserialize   Messages are logically partitioned based on the pipeline config's `pipeline_batch_size` parameter.
  dropna        Drop null data entries from a DataFrame.
  filter        Filter message by a classification threshold.
  from-doca     A source stage used to receive raw packet data from a ConnectX-6 Dx NIC.
  from-file     Load messages from a file.
  from-kafka    Load messages from a Kafka cluster.
  gen-viz       (Deprecated) Write out visualization DataFrames.
  inf-identity  Perform inference for testing that performs a no-op.
  inf-pytorch   Perform inference with PyTorch.
  inf-triton    Perform inference with Triton Inference Server.
  mlflow-drift  Report model drift statistics to ML Flow.
  monitor       Display throughput numbers at a specific point in the pipeline.
  preprocess    Prepare NLP input DataFrames for inference.
  serialize     Includes & excludes columns from messages.
  to-file       Write all messages to a file.
  to-kafka      Write all messages to a Kafka cluster.
  trigger       Buffer data until the previous stage has completed.
  validate      Validate pipeline output for testing.
```

And for the FIL pipeline:
```
$ morpheus run pipeline-fil --help
Usage: morpheus run pipeline-fil [OPTIONS] COMMAND1 [ARGS]... [COMMAND2 [ARGS]...]...

<Help Paragraph Omitted>

Commands:
  add-class       Add detected classifications to each message.
  add-scores      Add probability scores to each message.
  buffer          (Deprecated) Buffer results.
  delay           (Deprecated) Delay results for a certain duration.
  deserialize     Messages are logically partitioned based on the pipeline config's `pipeline_batch_size` parameter.
  dropna          Drop null data entries from a DataFrame.
  filter          Filter message by a classification threshold.
  from-appshield  Source stage is used to load Appshield messages from one or more plugins into a dataframe. It normalizes nested json messages and arranges them
                  into a dataframe by snapshot and source.
  from-file       Load messages from a file.
  from-kafka      Load messages from a Kafka cluster.
  inf-identity    Perform inference for testing that performs a no-op.
  inf-pytorch     Perform inference with PyTorch.
  inf-triton      Perform inference with Triton Inference Server.
  mlflow-drift    Report model drift statistics to ML Flow.
  monitor         Display throughput numbers at a specific point in the pipeline.
  preprocess      Prepare FIL input DataFrames for inference.
  serialize       Includes & excludes columns from messages.
  to-file         Write all messages to a file.
  to-kafka        Write all messages to a Kafka cluster.
  trigger         Buffer data until the previous stage has completed.
  validate        Validate pipeline output for testing.
```

And for the AE pipeline:

```
$ morpheus run pipeline-ae --help
Usage: morpheus run pipeline-ae [OPTIONS] COMMAND1 [ARGS]... [COMMAND2 [ARGS]...]...

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
  serialize        Includes & excludes columns from messages.
  timeseries       Perform time series anomaly detection and add prediction.
  to-file          Write all messages to a file.
  to-kafka         Write all messages to a Kafka cluster.
  train-ae         Train an Autoencoder model on incoming data.
  trigger          Buffer data until the previous stage has completed.
  validate         Validate pipeline output for testing.
```
Note: The available commands for different types of pipelines are not the same. This means that the same stage, when used in different pipelines, may have different options. Check the CLI help for the most up-to-date information during development.

## Next Steps
* [Morpheus Examples](./examples.md) - Example pipelines using both the Python API and command line interface
* [Morpheus Pretrained Models](./models_and_datasets.md) - Pre-trained models with corresponding training, validation scripts, and datasets
* [Morpheus Developer Guide](./developer_guide/guides.md) - Documentation on using the Morpheus Python & C++ APIs
