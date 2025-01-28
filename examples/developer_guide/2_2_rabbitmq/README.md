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
This example includes two stages `RabbitMQSourceStage` and `WriteToRabbitMQStage`

## Supported Environments
| Environment | Supported | Notes |
|-------------|-----------|-------|
| Conda | ✔ | |
| Morpheus Docker Container | ✔ | Requires launching the RabbitMQ container on the host |
| Morpheus Release Container | ✔ | Requires launching the RabbitMQ container on the host |
| Dev Container | ✘ |  |


## Testing with a RabbitMQ container
Testing can be performed locally with the RabbitMQ supplied docker image from the [RabbitMQ container registry](https://registry.hub.docker.com/_/rabbitmq/):
```bash
docker run --rm -it --hostname my-rabbit -p 15672:15672 -p 5672:5672 rabbitmq:3-management
```

The image can be verified with the web management console by opening http://localhost:15672 in a web browser. Enter "guest" for both the username and the password.

## Installing Pika
The `RabbitMQSourceStage` and `WriteToRabbitMQStage` stages use the [pika](https://pika.readthedocs.io/en/stable/#) RabbitMQ client for Python. To install this into the current environment run:
```bash
pip install -r examples/developer_guide/2_2_rabbitmq/requirements.txt
```

## Launch the reader
In a second terminal from the root of the Morpheus repo execute:
```bash
export MORPHEUS_ROOT=$(pwd)
python examples/developer_guide/2_2_rabbitmq/read_simple.py
```

This will read from a RabbitMQ exchange named 'logs', and write the results to `results.json`.

If no exchange named 'logs' exists in RabbitMQ it will be created. By default the `read_simple.py` script will utilize the class-based `RabbitMQSourceStage`, alternately using the `--use_source_function` flag will utilize the function-based `rabbitmq_source` stage.

## Launch the writer
In a third terminal from the root of the Morpheus repo execute:
```bash
export MORPHEUS_ROOT=$(pwd)
python examples/developer_guide/2_2_rabbitmq/write_simple.py
```

This will read JSON data from the `examples/data/email.jsonlines` file and publish the data into the 'logs' RabbitMQ exchange as a single message.

The `write_simple.py` script will exit as soon as the message is written to the queue. The `read_simple.py` script will continue reading from the queue until explicitly shut down with a control-C.

> **Note**: Both the `read_simple.py` and `write_simple.py` scripts will launch independent Morpheus pipelines, both of which can optionally execute in CPU-only mode by setting the `--use_cpu_only` flag.

## Alternate Morpheus CLI usage
In the above examples we defined the pipeline using the Python API in the `read_simple.py` and `write_simple.py` scripts. Alternately, we could have defined the same pipelines using the Morpheus CLI tool.

### Read Pipeline
From the  Morpheus repo root directory run:
```bash
export MORPHEUS_ROOT=$(pwd)
morpheus --log_level=INFO --plugin examples/developer_guide/2_2_rabbitmq/rabbitmq_source_stage.py \
  run pipeline-other \
  from-rabbitmq --host=localhost --exchange=logs \
  monitor \
  to-file --filename=results.json --overwrite
```

### Write Pipeline
From the  Morpheus repo root directory run:
```bash
export MORPHEUS_ROOT=$(pwd)
morpheus --log_level=INFO --plugin examples/developer_guide/2_2_rabbitmq/write_to_rabbitmq_stage.py \
  run pipeline-other \
  from-file --filename=examples/data/email.jsonlines \
  to-rabbitmq --host=localhost --exchange=logs
```
