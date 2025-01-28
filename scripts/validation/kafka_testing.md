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

This document walks through manual testing of the Kafka functionality in Morpheus. There are also several automated tests which are run as part of the CI process. To run the tests locally we will need to install a few dependencies needed for the tests:
```bash
mamba install -c conda-forge "openjdk=11.0.15"
npm install -g camouflage-server@0.15
python ${MORPHEUS_ROOT}/ci/scripts/download_kafka.py
cd ${MORPHEUS_ROOT}
```

Then run the Kafka tests with:
```bash
pytest --run_slow --run_kafka
```

## Pre-reqs
1. Create the `${MORPHEUS_ROOT}/.tmp` dir (this dir is already listed in the `.gitignore` file).
    ```bash
    mkdir -p ${MORPHEUS_ROOT}/.tmp
    ```
1. To help validate the data we will be using the `jq` command, if this is not already installed on your system it can be installed with:
    ```bash
    mamba install -c conda-forge jq
    ```
1. Launch Kafka using instructions from the [Quick Launch Kafka Cluster](../../docs/source/developer_guide/contributing.md#quick-launch-kafka-cluster) section of [contributing.md](../../docs/source/developer_guide/contributing.md) following steps 1-6.

1. The testing steps below will require two separate terminal windows. Each will need to have the `KAFKA_ADVERTISED_HOST_NAME`, `BROKER_LIST` and `MORPHEUS_ROOT` environment variables set. In the example below both `morpheus` and `kafka-docker` repositories have been checked out into the `~work` directory, replacing these paths with the location of your checkouts.
    ```bash
    export MORPHEUS_ROOT=~/work/morpheus
    export KAFKA_ADVERTISED_HOST_NAME=$(docker network inspect bridge | jq -r '.[0].IPAM.Config[0].Gateway')
    export BROKER_LIST=$(HOST_IP=$KAFKA_ADVERTISED_HOST_NAME ~/work/kafka-docker/broker-list.sh)
    ```
1. Open a new terminal and start the Kafka docker container:
    ```bash
    docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock \
         -e HOST_IP=$KAFKA_ADVERTISED_HOST_NAME -e ZK=$2 \
         -v ${MORPHEUS_ROOT}:/workspace wurstmeister/kafka /bin/bash
    ```

    Leave this terminal open the testing steps will refer to these as the "Kafka terminal" and commands executed from this terminal will be within the Kafka container.

1. Open a new terminal and navigate to the root of the Morpheus repo, this will be referred to as the "Morpheus terminal" and will be used for running Morpheus pipelines and verifying output.

### File descriptors
If you receive errors from Kafka such as `Too many open files`, you may need to increase the maximum number of open file descriptors. To check the current file descriptor limit run:
```bash
ulimit -n
```

To increase the limit (in this example to `4096`):
```bash
ulimit -n 4096
```

## Simple Data Copying
### Checking KafkaSourceStage
#### Single Partition Topic Test
1. From the Kafka terminal, create a topic called `morpheus-src-copy-test` with only a single partition.
    ```bash
    $KAFKA_HOME/bin/kafka-topics.sh --create --topic=morpheus-src-copy-test  --partitions 1 --bootstrap-server `broker-list.sh`
    ```

1. From the Morpheus terminal launch a pipeline to listen to Kafka:
    ```bash
    morpheus --log_level=DEBUG run \
        pipeline-nlp \
        from-kafka --input_topic morpheus-src-copy-test --bootstrap_servers "${BROKER_LIST}" \
        monitor --description "Kafka Read" \
        deserialize \
        serialize \
        to-file --include-index-col=false --filename=${MORPHEUS_ROOT}/.tmp/morpheus-src-copy-test.csv --overwrite
    ```

1. Return to the Kafka terminal and run:
    ```bash
    cat /workspace/tests/tests_data/filter_probs.jsonlines | \
        $KAFKA_HOME/bin/kafka-console-producer.sh \
        --topic=morpheus-src-copy-test --broker-list=`broker-list.sh` -
    ```

1. Return to the Morpheus terminal, and once the monitor stage has recorded: `read: 20 messages` shut down the pipeline with Ctrl-C.

1. If successful the output file `.tmp/morpheus-src-copy-test.csv` should be identical to `tests/tests_data/filter_probs.csv`. Verify:
    ```bash
    diff -q --ignore-all-space ${MORPHEUS_ROOT}/tests/tests_data/filter_probs.csv ${MORPHEUS_ROOT}/.tmp/morpheus-src-copy-test.csv
    ```

#### Partitioned Topic Test
1. From the Kafka terminal create a new topic named `morpheus-src-copy-test-p` with three partitions:
    ```bash
    $KAFKA_HOME/bin/kafka-topics.sh --create --topic=morpheus-src-copy-test-p --partitions 3 --bootstrap-server `broker-list.sh`
    ```

1. From the Morpheus terminal run:
    ```bash
    morpheus --log_level=DEBUG run \
        pipeline-nlp \
        from-kafka --input_topic morpheus-src-copy-test-p --bootstrap_servers "${BROKER_LIST}" \
        deserialize \
        monitor --description "Kafka Read" \
        serialize \
        to-file --include-index-col=false --filename=${MORPHEUS_ROOT}/.tmp/morpheus-src-copy-test-p.csv --overwrite
    ```

1. Return to the Kafka terminal and run:
    ```bash
    cat /workspace/tests/tests_data/filter_probs.jsonlines | \
        $KAFKA_HOME/bin/kafka-console-producer.sh \
        --topic=morpheus-src-copy-test-p --broker-list=`broker-list.sh` -
    ```

1. Return to the Morpheus terminal, and once the monitor stage has recorded: `read: 20 messages` shut down the pipeline with Ctrl-C.

1. If successful the output file `.tmp/morpheus-src-copy-test-p.csv` should contain the same records as those in `tests/tests_data/filter_probs.csv` however they are most likely out of order. To verify the output we will compare the sorted outputs:
    ```bash
    diff -q --ignore-all-space <(sort tests/tests_data/filter_probs.csv) <(sort .tmp/morpheus-src-copy-test-p.csv)
    ```


### Checking WriteToKafkaStage
#### Single Partition Topic Test
1. From the Kafka terminal create a topic called `morpheus-sink-copy-test` with only a single partition, and start a consumer on that topic:
    ```bash
    $KAFKA_HOME/bin/kafka-topics.sh --create --topic=morpheus-sink-copy-test  --partitions 1 --bootstrap-server `broker-list.sh`

    $KAFKA_HOME/bin/kafka-console-consumer.sh --topic=morpheus-sink-copy-test \
        --bootstrap-server `broker-list.sh` > /workspace/.tmp/morpheus-sink-copy-test.jsonlines
    ```

1. From the Morpheus terminal run:
    ```bash
    morpheus --log_level=DEBUG run \
        pipeline-nlp \
        from-file --filename=${MORPHEUS_ROOT}/tests/tests_data/filter_probs.csv \
        deserialize  \
        serialize \
        to-kafka --output_topic morpheus-sink-copy-test --bootstrap_servers "${BROKER_LIST}"
    ```
    The `tests/tests_data/filter_probs.csv` contains 20 lines of data and the pipeline should complete rather quickly (less than 5 seconds).

1. The Kafka consumer we started in step #1 won't give us any sort of indication as to how many records have been consumed, we will indirectly check the progress by counting the rows in the output file. Once the Morpheus pipeline completes check the number of lines in the output:
    ```bash
    wc -l ${MORPHEUS_ROOT}/.tmp/morpheus-sink-copy-test.jsonlines
    ```

1. Once all 20 lines have been written to the output file, verify the contents with:
    ```bash
    diff -q --ignore-all-space <(cat ${MORPHEUS_ROOT}/.tmp/morpheus-sink-copy-test.jsonlines | jq --sort-keys) <(cat ${MORPHEUS_ROOT}/tests/tests_data/filter_probs.jsonlines | jq --sort-keys)
    ```
    Note the usage of `jq --sort-keys` which will reformat the JSON output, sorting the keys, this ensures that `{"a": 5, "b": 6}` and `{"b": 6,   "a": 5}` are considered equivalent.

1. Stop the consumer in the Kafka terminal.

#### Partitioned Topic Test
1. From the Kafka terminal create a new topic named `morpheus-sink-copy-test-p` with three partitions, and start a consumer on that topic:
    ```bash
    $KAFKA_HOME/bin/kafka-topics.sh --create --topic=morpheus-sink-copy-test-p --partitions 3 --bootstrap-server `broker-list.sh`

    $KAFKA_HOME/bin/kafka-console-consumer.sh --topic=morpheus-sink-copy-test-p \
        --bootstrap-server `broker-list.sh` > /workspace/.tmp/morpheus-sink-copy-test-p.jsonlines
    ```

1. From the Morpheus terminal run:
    ```bash
    morpheus --log_level=DEBUG run \
        pipeline-nlp \
        from-file --filename=${MORPHEUS_ROOT}/tests/tests_data/filter_probs.csv \
        deserialize  \
        serialize \
        to-kafka --output_topic morpheus-sink-copy-test-p --bootstrap_servers "${BROKER_LIST}"
    ```
    The `tests/tests_data/filter_probs.csv` contains 20 lines of data and the pipeline should complete rather quickly (less than 5 seconds).

1. The Kafka consumer we started in step #1 won't give us any sort of indication that it has concluded we will indirectly check the progress by counting the rows in the output file. Once the Morpheus pipeline completes check the number of lines in the output:
    ```bash
    wc -l ${MORPHEUS_ROOT}/.tmp/morpheus-sink-copy-test-p.jsonlines
    ```

1. Once all 20 lines have been written to the output file, verify the contents with:
    ```bash
    diff -q --ignore-all-space <(sort ${MORPHEUS_ROOT}/.tmp/morpheus-sink-copy-test-p.jsonlines | jq --sort-keys) <(sort ${MORPHEUS_ROOT}/tests/tests_data/filter_probs.jsonlines | jq --sort-keys)
    ```
    Note due to the multiple partitions the consumer most likely received records out of order, so we are comparing the sorted output of both files.

1. Stop the consumer in the Kafka terminal.


## Optional Cleanup
### Delete all topics
1. Return to the Kafka terminal and within the container run:
    ```bash
    $KAFKA_HOME/bin/kafka-topics.sh --list --bootstrap-server `broker-list.sh` | xargs -I'{}' $KAFKA_HOME/bin/kafka-topics.sh --delete --bootstrap-server `broker-list.sh` --topic='{}'
    ```

    Note: The Kafka containers are using a persistent volume, and the topics will persist after a restart of the docker containers.

### Shutdown Kafka
1. Exit the Kafka terminal.
1. From the root of the `kafka-docker` repo run (in the host OS not inside a container):
    ```bash
    docker compose stop
    docker compose rm
    ```
