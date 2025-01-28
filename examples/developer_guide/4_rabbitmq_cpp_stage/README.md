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

# Example RabbitMQ stages
This example builds upon the `examples/developer_guide/2_2_rabbitmq` example adding a C++ implementation for the `RabbitMQSourceStage` along with adding package install scripts.

## Supported Environments
| Environment | Supported | Notes |
|-------------|-----------|-------|
| Conda | ✔ | |
| Morpheus Docker Container | ✔ | Requires launching the RabbitMQ container on the host |
| Morpheus Release Container | ✔ | Requires launching the RabbitMQ container on the host, and adding development packages to the container's Conda environment via `conda env update --solver=libmamba -n morpheus --file /workspace/conda/environments/dev_cuda-125_arch-$(arch).yaml` |
| Dev Container | ✘ |  |

## Installing Pika
The `RabbitMQSourceStage` and `WriteToRabbitMQStage` stages use the [pika](https://pika.readthedocs.io/en/stable/#) RabbitMQ client for Python. To install this into the current environment run:
```bash
pip install -r examples/developer_guide/4_rabbitmq_cpp_stage/requirements.txt
```

## Building the Example
There are two ways to build the example. The first is to build the examples along with Morpheus by passing the `-DMORPHEUS_BUILD_EXAMPLES=ON` flag to CMake, for users using the `scripts/compile.sh` at the root of the Morpheus repo can do this by setting the `CMAKE_CONFIGURE_EXTRA_ARGS` environment variable:
```bash
CMAKE_CONFIGURE_EXTRA_ARGS="-DMORPHEUS_BUILD_EXAMPLES=ON" ./scripts/compile.sh
```

The second is to build the example as a standalone project. From the root of the Morpheus repo execute:
```bash
cd examples/developer_guide/4_rabbitmq_cpp_stage
./compile.sh

# Optionally install the package into the current python environment
pip install ./
```

## Testing with a RabbitMQ container
Testing can be performed locally with the RabbitMQ supplied docker image from the [RabbitMQ container registry](https://registry.hub.docker.com/_/rabbitmq/):
```bash
docker run --rm -it --hostname my-rabbit -p 15672:15672 -p 5672:5672 rabbitmq:3-management
```

The image can be verified with the web management console by opening http://localhost:15672 in a web browser. Enter "guest" for both the username and the password.

## Launch the reader
In a second terminal from the root of the Morpheus repo execute:
```bash
export MORPHEUS_ROOT=$(pwd)
python examples/developer_guide/4_rabbitmq_cpp_stage/src/read_simple.py
```

This will read from a RabbitMQ exchange named 'logs', and write the results to `results.json`.

If no exchange named 'logs' exists in RabbitMQ it will be created.

## Launch the writer
In a third terminal from the root of the Morpheus repo execute:
```bash
export MORPHEUS_ROOT=$(pwd)
python examples/developer_guide/4_rabbitmq_cpp_stage/src/write_simple.py
```

This will read JSON data from the `examples/data/email.jsonlines` file and publish the data into the 'logs' RabbitMQ exchange as a single message.

The `write_simple.py` script will exit as soon as the message is written to the queue. The `read_simple.py` script will continue reading from the queue until explicitly shut down with a control-C.
