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

# Morpheus Cloud Deployment Guide

## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Set up NGC API Key and Install NGC Registry CLI](#set-up-ngc-api-key-and-install-ngc-registry-cli)
  - [Create Namespace for Morpheus](#create-namespace-for-morpheus)
  - [Install Morpheus AI Engine](#install-morpheus-ai-engine)
  - [Install Morpheus SDK Client](#install-morpheus-sdk-client)
    - [Morpheus SDK Client in Sleep Mode](#morpheus-sdk-client-in-sleep-mode)
  - [Models for MLflow Deployment](#models-for-mlflow-deployment)
  - [Install Morpheus MLflow](#install-morpheus-mlflow)
  - [Model Deployment](#model-deployment)
  - [Verify Model Deployment](#verify-model-deployment)
  - [Create Kafka Topics](#create-kafka-topics)
- [Example Workflows](#example-workflows)
  - [Run NLP Phishing Detection Pipeline](#run-nlp-phishing-detection-pipeline)
  - [Run NLP Sensitive Information Detection Pipeline](#run-nlp-sensitive-information-detection-pipeline)
  - [Run FIL Anomalous Behavior Profiling Pipeline](#run-fil-anomalous-behavior-profiling-pipeline)
  - [Verify Running Pipeline](#verify-running-pipeline)
- [Prerequisites and Installation for AWS](#prerequisites-and-installation-for-aws)
  - [Prerequisites](#prerequisites-1)
  - [Install Cloud Native Core Stack for AWS](#install-cloud-native-core-stack-for-aws)
- [Prerequisites and Installation for Ubuntu](#prerequisites-and-installation-for-ubuntu)
  - [Prerequisites](#prerequisites-2)
- [Installing Cloud Native Core Stack on NVIDIA Certified Systems](#installing-cloud-native-core-stack-on-nvidia-certified-systems)
- [Kafka Topic Commands](#kafka-topic-commands)
- [Additional Documentation](#additional-documentation)
- [Troubleshooting](#troubleshooting)
  - [Common Problems](#common-problems)


## Introduction

This cloud deployment guide provides the necessary instructions to set up the minimum infrastructure and configuration needed to deploy the Morpheus Developer Kit and includes example workflows leveraging the deployment.

- This cloud deployment guide consists of the following steps:
- Set up of the NVIDIA Cloud Native Core Stack
- Set up Morpheus AI Engine
- Set up Morpheus SDK Client
- Models for MLflow Deployment
- Set up Morpheus MLflow
- Deploy models to Triton inference server
- Create Kafka topics
- Run example workloads

> **Note**: This guide requires access to the NGC Public Catalog.

## Setup

### Prerequisites
1.  Refer to prerequisites for Cloud (AWS) [here](#prerequisites-1) or On-Prem (Ubuntu) [here](#prerequisites-2)
2.  Registration in the NGC Public Catalog

Continue with the setup steps below once the host system is installed, configured, and satisfies all prerequisites.

### Set up NGC API Key and Install NGC Registry CLI

First, you will need to set up your NGC API Key to access all the Morpheus components, using the linked instructions from the [NGC Registry CLI User Guide](https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html#generating-personal-api-key).

Once you've created your API key, create an environment variable containing your API key for use by the commands used further in this document:

```bash
export API_KEY="<NGC_API_KEY>"
```

Next, install and configure the NGC Registry CLI on your system using the linked instructions from the [NGC Registry CLI User Guide](https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html#generating-personal-api-key).

### Create Namespace for Morpheus

Next, create a namespace and an environment variable for the namespace to organize the Kubernetes cluster deployed via the Cloud Native Core Stack and logically separate Morpheus related deployments from other projects using the following command:

```bash
export NAMESPACE="<YOUR_NAMESPACE>"
kubectl create namespace ${NAMESPACE}
```

### Install Morpheus AI Engine

The Helm chart (`morpheus-ai-engine`) that offers the auxiliary components required to execute certain Morpheus workflows is referred to as the Morpheus AI Engine. It comprises of the following components
-   Triton Inference Server [ **ai-engine** ] from NVIDIA for processing inference requests.
-   Kafka Broker [ **broker** ] to consume and publish messages.
-   Zookeeper [ **zookeeper** ] to maintain coordination between the Kafka Brokers.

Follow the below steps to install Morpheus AI Engine:

```bash
helm fetch https://helm.ngc.nvidia.com/nvidia/morpheus/charts/morpheus-ai-engine-25.02.tgz --username='$oauthtoken' --password=$API_KEY --untar
```
```bash
helm install --set ngc.apiKey="$API_KEY" \
             --namespace $NAMESPACE \
             <YOUR_RELEASE_NAME> \
             morpheus-ai-engine
```

After the installation, you can verify that the Kubernetes pods are running successfully using the following command:

```bash
kubectl -n $NAMESPACE get all
```

Output:

```console
pod/ai-engine-65f59ddcf7-mdmdt   1/1     Running   0          54s
pod/broker-76f7c64dc9-6rldp      1/1     Running   1          54s
pod/zookeeper-87f9f4dd-znjnb     1/1     Running   0          54s

NAME                TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)                      AGE
service/ai-engine   ClusterIP   10.109.56.231    <none>        8000/TCP,8001/TCP,8002/TCP   54s
service/broker      ClusterIP   10.101.103.250   <none>        9092/TCP                     54s
service/zookeeper   ClusterIP   10.110.55.141    <none>        2181/TCP                     54s

NAME                        READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/ai-engine   1/1     1            1           54s
deployment.apps/broker      1/1     1            1           54s
deployment.apps/zookeeper   1/1     1            1           54s

NAME                                   DESIRED   CURRENT   READY   AGE
replicaset.apps/ai-engine-65f59ddcf7   1         1         1       54s
replicaset.apps/broker-76f7c64dc9      1         1         1       54s
replicaset.apps/zookeeper-87f9f4dd     1         1         1       54s
```

### Install Morpheus SDK Client
Run the following command to pull the Morpheus SDK Client (referred to as Helm chart `morpheus-sdk-client`) on to your instance:

```bash
helm fetch https://helm.ngc.nvidia.com/nvidia/morpheus/charts/morpheus-sdk-client-25.02.tgz --username='$oauthtoken' --password=$API_KEY --untar
```

#### Morpheus SDK Client in Sleep Mode
Install the Morpheus SDK client pod in sleep mode to copy its sample datasets and models from the container to a shared location that other pods can access. If no `sdk.args` is supplied, the default value `/bin/sleep infinity` from the chart is used in the following command.

```bash
helm install --set ngc.apiKey="$API_KEY" \
               --namespace $NAMESPACE \
               helper \
               morpheus-sdk-client
```

Check the status of the pod to make sure it's up and running.

```bash
kubectl -n $NAMESPACE get all | grep sdk-cli-helper
```

Output:

```console
pod/sdk-cli-helper           1/1     Running   0               41s
```

### Models for MLflow Deployment

Connect to the **sdk-cli-helper** container and copy the models to `/common`, which is mapped to `/opt/morpheus/common` on the host and where MLflow will have access to model files.

```bash
kubectl -n $NAMESPACE exec sdk-cli-helper -- cp -RL /workspace/models /common
```

### Install Morpheus MLflow

The Morpheus MLflow Helm chart offers MLflow server with Triton plugin to deploy, update, and remove models from the Morpheus AI Engine. The MLflow server UI can be accessed using NodePort `30500`. Follow the below steps to install the Morpheus MLflow:

```bash
helm fetch https://helm.ngc.nvidia.com/nvidia/morpheus/charts/morpheus-mlflow-25.02.tgz --username='$oauthtoken' --password=$API_KEY --untar
```
```bash
helm install --set ngc.apiKey="$API_KEY" \
             --namespace $NAMESPACE \
             <YOUR_RELEASE_NAME> \
             morpheus-mlflow
```

> **Note**: If the default port is already allocated, Helm throws below error. Choose an alternative by adjusting the `dashboardPort` value in the `morpheus-mlflow/values.yaml` file, remove the previous release and reinstall it.

```console
Error: Service "mlflow" is invalid: spec.ports[0].nodePort: Invalid value: 30500: provided port is already allocated
```

After the installation, you can verify that the MLflow pod is running successfully using the following command:

```bash
kubectl -n $NAMESPACE get all | grep  pod/mlflow
```

Output:
```bash
pod/mlflow-6d98        1/1     Running   0          39s
```

### Model Deployment
Attach to the MLflow pod to publish models to the MLflow server and then deploy it onto Morpheus AI Engine:

```bash
kubectl -n $NAMESPACE exec -it deploy/mlflow -- bash
```

```console
(mlflow) root@mlflow-6d98:/mlflow#
```

`Important`: When `(mlflow)` is present, commands are directly within the container.

First let's examine the syntax of the commands we will be using to communicate with the MLflow Triton plugin before we start deploying models.
Publish models to MLflow server is in the form of:

```bash
(mlflow) root@mlflow-6d98:/mlflow# python publish_model_to_mlflow.py \
      --model_name <REF_MODEL_NAME> \
      --model_directory <MODEL_DIR_PATH> \
      --flavor <MODEL_FLAVOR>
```

Deploy models to Morpheus AI Engine:

```bash
(mlflow) root@mlflow-6d98:/mlflow# mlflow deployments create -t triton \
     --flavor <MODEL_FLAVOR> \
     --name <REF_MODEL_NAME> \
     -m models:/<REF_MODEL_NAME>/1 \
     -C "version=<VERSION_NUMBER>"
```

Update deployed models in Morpheus AI Engine:

```
(mlflow) root@mlflow-6d98:/mlflow# mlflow deployments update -t triton \
     --flavor <MODEL_FLAVOR> \
     --name <REF_MODEL_NAME>/<EXISTING_VERSION_NUMBER> \
     -m models:/<REF_MODEL_NAME>/<DESIRED_VERSION_NUMBER>
```

Delete deployed models from Morpheus AI Engine:

```bash
(mlflow) root@mlflow-6d98:/mlflow# mlflow deployments delete -t triton \
     --name <REF_MODEL_NAME>/<VERSION_NUMBER>
```

Now that we've figured out how to deploy models let's move on to the next step. Now it's time to deploy the relevant models, which have already been copied to `/opt/morpheus/common/models` which are bound to `/common/models` within the MLflow pod.

```bash
(mlflow) root@mlflow-6d98:/mlflow# ls -lrt /common/models
```

Output:

```console
drwxr-xr-x 3 ubuntu ubuntu 4096 Apr 13 23:47 sid-minibert-onnx
drwxr-xr-x 2 root   root   4096 Apr 21 17:09 abp-models
drwxr-xr-x 4 root   root   4096 Apr 21 17:09 datasets
drwxr-xr-x 4 root   root   4096 Apr 21 17:09 fraud-detection-models
drwxr-xr-x 2 root   root   4096 Apr 21 17:09 dfp-models
drwxr-xr-x 3 root   root   4096 Apr 21 17:10 mlflow
drwxr-xr-x 2 root   root   4096 Apr 21 17:10 log-parsing-models
drwxr-xr-x 2 root   root   4096 Apr 21 17:10 phishing-models
drwxr-xr-x 2 root   root   4096 Apr 21 17:10 sid-models
drwxr-xr-x 8 root   root   4096 Apr 21 17:10 training-tuning-scripts
drwxr-xr-x 7 root   root   4096 Apr 21 17:10 validation-inference-scripts
drwxr-xr-x 7 root   root   4096 Apr 21 17:10 triton-model-repo
-rw-r--r-- 1 root   root   4213 Apr 21 17:10 README.md
-rw-r--r-- 1 root   root   4862 Apr 21 17:10 model_cards.csv
-rw-r--r-- 1 root   root   1367 Apr 21 17:10 model-information.csv
```


Publish and deploy sid-minibert-onnx model:

```bash
(mlflow) root@mlflow-6d98:/mlflow# python publish_model_to_mlflow.py \
      --model_name sid-minibert-onnx \
      --model_directory /common/models/triton-model-repo/sid-minibert-onnx \
      --flavor triton
```

```bash
(mlflow) root@mlflow-6d98:/mlflow# mlflow deployments create -t triton \
      --flavor triton \
      --name sid-minibert-onnx \
      -m models:/sid-minibert-onnx/1 \
      -C "version=1"
```

Publish and deploy phishing-bert-onnx model:

```bash
(mlflow) root@mlflow-6d98:/mlflow# python publish_model_to_mlflow.py \
      --model_name phishing-bert-onnx \
      --model_directory /common/models/triton-model-repo/phishing-bert-onnx \
      --flavor triton
```
```bash
(mlflow) root@mlflow-6d98:/mlflow# mlflow deployments create -t triton \
      --flavor triton \
      --name phishing-bert-onnx \
      -m models:/phishing-bert-onnx/1 \
      -C "version=1"
```

Publish and deploy abp-nvsmi-xgb model:

```bash
(mlflow) root@mlflow-6d98:/mlflow# python publish_model_to_mlflow.py \
      --model_name abp-nvsmi-xgb \
      --model_directory /common/models/triton-model-repo/abp-nvsmi-xgb \
      --flavor triton
```

```bash
(mlflow) root@mlflow-6d98:/mlflow# mlflow deployments create -t triton \
      --flavor triton \
      --name abp-nvsmi-xgb \
      -m models:/abp-nvsmi-xgb/1 \
      -C "version=1"
```

Exit from the container

```bash
(mlflow) root@mlflow-6d98:/mlflow# exit
```

### Verify Model Deployment
Run the following command to verify that the models were successfully deployed on the AI Engine:

```bash
kubectl -n $NAMESPACE logs deploy/ai-engine
```

Output:
```console
I1202 14:09:03.098085 1 api.cu:79] TRITONBACKEND_ModelInitialize: abp-nvsmi-xgb (version 1)
I1202 14:09:03.101910 1 api.cu:123] TRITONBACKEND_ModelInstanceInitialize: abp-nvsmi-xgb_0 (GPU device 0)
I1202 14:09:03.543719 1 model_instance_state.cu:101] Using GPU for predicting with model 'abp-nvsmi-xgb_0'
I1202 14:09:03.563425 1 api.cu:123] TRITONBACKEND_ModelInstanceInitialize: abp-nvsmi-xgb_0 (GPU device 1)
I1202 14:09:03.980621 1 model_instance_state.cu:101] Using GPU for predicting with model 'abp-nvsmi-xgb_0'
I1202 14:09:03.981678 1 model_repository_manager.cc:1183] successfully loaded 'abp-nvsmi-xgb' version 1
```

### Create Kafka Topics
We will need to create Kafka topics for input and output data to run some of the pipeline examples.

Check if any Kafka topics exist already. If any exist, you can either delete the previous topics or re-use them.

```bash
kubectl -n $NAMESPACE exec deploy/broker -c broker -- kafka-topics.sh --list  --zookeeper zookeeper:2181
```

Run the following command twice, once to create an input topic, and again to create an output topic, making sure that the input topic and output topic have different names:

```bash
kubectl -n $NAMESPACE exec deploy/broker -c broker -- kafka-topics.sh \
      --create \
      --bootstrap-server broker:9092 \
      --replication-factor 1 \
      --partitions 3 \
      --topic <YOUR_KAFKA_TOPIC>
```

## Example Workflows

This section describes example workflows to run on Morpheus. Four sample pipelines are provided.

1. NLP pipeline performing Phishing Detection (PD).
2. NLP pipeline performing Sensitive Information Detection (SID).
3. FIL pipeline performing Anomalous Behavior Profiling (ABP).

Multiple command options are given for each pipeline, with varying data input/output methods, ranging from local files to Kafka Topics.

We recommend only deploying one pipeline at a time. To remove previously deployed pipelines, run the following command:

```bash
helm delete -n $NAMESPACE <YOUR_RELEASE_NAME>
```

To publish messages to a Kafka topic, we need to copy datasets to locations where they can be accessed from the host.

```bash
kubectl -n $NAMESPACE exec sdk-cli-helper -- cp -R /workspace/examples/data /common
```

Refer to the [Morpheus CLI Overview](./basics/overview.rst) and [Building a Pipeline](./basics/building_a_pipeline.md) documentation for more information regarding the commands.

> **Note**: Before running the example pipelines, ensure the criteria below are met:
-   Ensure models specific to the pipeline are deployed.
-   Input and Output Kafka topics have been created.
-   Recommended to create an output directory under  `/opt/morpheus/common/data` which is bound to `/common/data` (pod/container) for storing inference or validation results.
-   Replace **<YOUR_OUTPUT_DIR>** with your directory name.
-   Replace **<YOUR_INPUT_KAFKA_TOPIC>** with your input Kafka topic name.
-   Replace **<YOUR_OUTPUT_KAFKA_TOPIC>** with your output Kafka topic name.
-   Replace **<YOUR_RELEASE_NAME>** with the name you want.


For reference, the Morpheus SDK Client install pipeline command template is provided. We will examine this further in the [example workflows](#example-workflows) section, but for now, let's proceed to the next step.

```bash
helm install --set ngc.apiKey="$API_KEY" \
               --set sdk.args="<REPLACE_RUN_PIPELINE_COMMAND_HERE>" \
               --namespace $NAMESPACE \
               <YOUR_RELEASE_NAME> \
               morpheus-sdk-client
```

### Run NLP Phishing Detection Pipeline

The following Phishing Detection pipeline examples use a pre-trained NLP model to analyze emails (body) and determine phishing or benign. Here is the sample data as shown below is used to pass as an input to the pipeline.

```json
{"data":"Abedin Huma <AbedinH@state.gov>Wednesday July 15 2009 1:44 PMRe: ArtWill be off campus at meetingBut you should definitely come I think they have found some good things."}
{"data":"See NIMills Cheryl D <MillsCD@state.gov>Saturday December 112010 1:36 PMFw: S is calling Leahy today - thx for all the help; advise if a diff no for him today"}
{"data":"Here is Draft"}
{"data":"Ok"}
```

Pipeline example to read data from a file, run inference using a `phishing-bert-onnx` model, and write inference results to the specified output file:

```bash
helm install --set ngc.apiKey="$API_KEY" \
    --set sdk.args="morpheus --log_level=DEBUG run \
      --edge_buffer_size=4 \
      --pipeline_batch_size=1024 \
      --model_max_batch_size=32 \
      pipeline-nlp \
        --model_seq_length=128 \
        --labels_file=data/labels_phishing.txt \
        from-file --filename=./examples/data/email.jsonlines \
        monitor --description 'FromFile Rate' --smoothing=0.001 \
        deserialize \
        preprocess --vocab_hash_file=data/bert-base-uncased-hash.txt --truncation=True --do_lower_case=True --add_special_tokens=False \
        monitor --description 'Preprocess Rate' \
        inf-triton --model_name=phishing-bert-onnx --server_url=ai-engine:8000 --force_convert_inputs=True \
        monitor --description 'Inference Rate' --smoothing=0.001 --unit inf \
        add-class --label=is_phishing --threshold=0.7 \
        serialize \
        to-file --filename=/common/data/<YOUR_OUTPUT_DIR>/phishing-bert-onnx-output.jsonlines --overwrite" \
    --namespace $NAMESPACE \
    <YOUR_RELEASE_NAME> \
    morpheus-sdk-client
```

When the pipeline runs successfully, an output file `phishing-bert-onnx-output.jsonlines` will appear in the output directory.

Pipeline example to read messages from an input Kafka topic, run inference using a   `phishing-bert-onnx`  model, and write the results of the inference to an output Kafka topic:

```bash
helm install --set ngc.apiKey="$API_KEY" \
    --set sdk.args="morpheus --log_level=DEBUG run \
      --edge_buffer_size=4 \
      --pipeline_batch_size=1024 \
      --model_max_batch_size=32 \
      pipeline-nlp \
        --model_seq_length=128 \
        --labels_file=data/labels_phishing.txt \
        from-kafka --input_topic <YOUR_INPUT_KAFKA_TOPIC> --bootstrap_servers broker:9092 \
        monitor --description 'FromKafka Rate' --smoothing=0.001 \
        deserialize \
        preprocess --vocab_hash_file=data/bert-base-uncased-hash.txt --truncation=True --do_lower_case=True --add_special_tokens=False \
        monitor --description 'Preprocess Rate' \
        inf-triton --force_convert_inputs=True --model_name=phishing-bert-onnx --server_url=ai-engine:8000 \
        monitor --description='Inference Rate' --smoothing=0.001 --unit inf \
        add-class --label=is_phishing --threshold=0.7 \
        serialize --exclude '^ts_' \
        to-kafka --output_topic <YOUR_OUTPUT_KAFKA_TOPIC> --bootstrap_servers broker:9092" \
    --namespace $NAMESPACE \
    <YOUR_RELEASE_NAME> \
    morpheus-sdk-client
```

Make sure you create input and output Kafka topics before you start the pipeline. After the pipeline has been started, load the individual corresponding data files from the downloaded sample into the selected input topic using the command below:

```bash
kubectl -n $NAMESPACE exec -it deploy/broker -c broker -- kafka-console-producer.sh \
       --broker-list broker:9092 \
       --topic <YOUR_INPUT_KAFKA_TOPIC> < \
       <YOUR_INPUT_DATA_FILE_PATH_EXAMPLE: /opt/morpheus/common/data/email.jsonlines>
```

> **Note**: This should be used for development purposes only via this developer kit. Loading from the file into Kafka should not be used in production deployments of Morpheus.

### Run NLP Sensitive Information Detection Pipeline
The following Sensitive Information Detection pipeline examples use a pre-trained NLP model to ingest and analyze PCAP (packet capture network traffic) input sample data, like the example below, to inspect IP traffic across data center networks.

```json
{"timestamp": 1616380971990, "host_ip": "10.188.40.56", "data_len": "309", "data": "POST /simpledatagen/ HTTP/1.1\r\nHost: echo.gtc1.netqdev.cumulusnetworks.com\r\nUser-Agent: python-requests/2.22.0\r\nAccept-Encoding: gzip, deflate\r\nAccept: */*\r\nConnection: keep-alive\r\nContent-Length: 73\r\nContent-Type: application/json\r\n\r\n", "src_mac": "04:3f:72:bf:af:74", "dest_mac": "b4:a9:fc:3c:46:f8", "protocol": "6", "src_ip": "10.20.16.248", "dest_ip": "10.244.0.59", "src_port": "50410", "dest_port": "80", "flags": "24", "is_pii": false}
{"timestamp": 1616380971991, "host_ip": "10.188.40.56", "data_len": "139", "data": "\"{\\\"markerEmail\\\": \\\"FuRLFaAZ identify benefit BneiMvCZ join 92694759\\\"}\"", "src_mac": "04:3f:72:bf:af:74", "dest_mac": "b4:a9:fc:3c:46:f8", "protocol": "6", "src_ip": "10.244.0.1", "dest_ip": "10.244.0.25", "src_port": "50410", "dest_port": "80", "flags": "24", "is_pii": false}
```

Pipeline example to read data from a file, run inference using a `sid-minibert-onnx` model, and write inference results to the specified output file:

```bash
helm install --set ngc.apiKey="$API_KEY" \
    --set sdk.args="morpheus --log_level=DEBUG run \
      --edge_buffer_size=4 \
      --pipeline_batch_size=1024 \
      --model_max_batch_size=32 \
      pipeline-nlp \
        --model_seq_length=256 \
        from-file --filename=./examples/data/pcap_dump.jsonlines \
        monitor --description 'FromFile Rate' --smoothing=0.001 \
        deserialize \
        preprocess --vocab_hash_file=data/bert-base-uncased-hash.txt --truncation=True --do_lower_case=True --add_special_tokens=False \
        monitor --description='Preprocessing rate' \
        inf-triton --force_convert_inputs=True --model_name=sid-minibert-onnx --server_url=ai-engine:8000 \
        monitor --description='Inference rate' --smoothing=0.001 --unit inf \
        add-class \
        serialize --exclude '^ts_' \
        to-file --filename=/common/data/<YOUR_OUTPUT_DIR>/sid-minibert-onnx-output.jsonlines --overwrite" \
    --namespace $NAMESPACE \
    <YOUR_RELEASE_NAME> \
    morpheus-sdk-client
```

When the pipeline runs successfully, an output file *sid-minibert-onnx-output.jsonlines* will appear in the output directory.

Pipeline example to read messages from an input Kafka topic, run inference using a *sid-minibert-onnx* model, and write the results of the inference to an output Kafka topic:

```bash
helm install --set ngc.apiKey="$API_KEY" \
    --set sdk.args="morpheus --log_level=DEBUG run \
        --edge_buffer_size=4 \
        --pipeline_batch_size=1024 \
        --model_max_batch_size=32 \
        pipeline-nlp \
          --model_seq_length=256 \
          from-kafka --input_topic <YOUR_INPUT_KAFKA_TOPIC> --bootstrap_servers broker:9092 \
          monitor --description 'FromKafka Rate' --smoothing=0.001 \
          deserialize \
          preprocess --vocab_hash_file=data/bert-base-uncased-hash.txt --truncation=True --do_lower_case=True --add_special_tokens=False \
          monitor --description='Preprocessing Rate' \
          inf-triton --force_convert_inputs=True --model_name=sid-minibert-onnx --server_url=ai-engine:8000 \
          monitor --description='Inference Rate' --smoothing=0.001 --unit inf \
          add-class \
          serialize --exclude '^ts_' \
          to-kafka --output_topic <YOUR_OUTPUT_KAFKA_TOPIC> --bootstrap_servers broker:9092" \
    --namespace $NAMESPACE \
    <YOUR_RELEASE_NAME> \
    morpheus-sdk-client
```

Make sure you create input and output Kafka topics before you start the pipeline. After the pipeline has been started, load the individual corresponding data files from the downloaded sample into the selected input topic using the command below:

```bash
kubectl -n $NAMESPACE exec -it deploy/broker -c broker -- kafka-console-producer.sh \
       --broker-list broker:9092 \
       --topic <YOUR_INPUT_KAFKA_TOPIC> < \
       <YOUR_INPUT_DATA_FILE_PATH_EXAMPLE: ${HOME}/examples/data/pcap_dump.jsonlines>
```

> **Note**: This should be used for development purposes only via this developer kit. Loading from the file into Kafka should not be used in production deployments of Morpheus.

### Run FIL Anomalous Behavior Profiling Pipeline
The following Anomalous Behavior Profiling pipeline examples use a pre-trained FIL model to ingest and analyze NVIDIA System Management Interface (`nvidia-smi`) logs, like the example below, as input sample data to identify cryptocurrency mining activity on GPU devices.

```json
{"nvidia_smi_log.gpu.pci.tx_util": "0 KB/s", "nvidia_smi_log.gpu.pci.rx_util": "0 KB/s", "nvidia_smi_log.gpu.fb_memory_usage.used": "3980 MiB", "nvidia_smi_log.gpu.fb_memory_usage.free": "12180 MiB", "nvidia_smi_log.gpu.bar1_memory_usage.total": "16384 MiB", "nvidia_smi_log.gpu.bar1_memory_usage.used": "11 MiB", "nvidia_smi_log.gpu.bar1_memory_usage.free": "16373 MiB", "nvidia_smi_log.gpu.utilization.gpu_util": "0 %", "nvidia_smi_log.gpu.utilization.memory_util": "0 %", "nvidia_smi_log.gpu.temperature.gpu_temp": "61 C", "nvidia_smi_log.gpu.temperature.gpu_temp_max_threshold": "90 C", "nvidia_smi_log.gpu.temperature.gpu_temp_slow_threshold": "87 C", "nvidia_smi_log.gpu.temperature.gpu_temp_max_gpu_threshold": "83 C", "nvidia_smi_log.gpu.temperature.memory_temp": "57 C", "nvidia_smi_log.gpu.temperature.gpu_temp_max_mem_threshold": "85 C", "nvidia_smi_log.gpu.power_readings.power_draw": "61.77 W", "nvidia_smi_log.gpu.clocks.graphics_clock": "1530 MHz", "nvidia_smi_log.gpu.clocks.sm_clock": "1530 MHz", "nvidia_smi_log.gpu.clocks.mem_clock": "877 MHz", "nvidia_smi_log.gpu.clocks.video_clock": "1372 MHz", "nvidia_smi_log.gpu.applications_clocks.graphics_clock": "1312 MHz", "nvidia_smi_log.gpu.applications_clocks.mem_clock": "877 MHz", "nvidia_smi_log.gpu.default_applications_clocks.graphics_clock": "1312 MHz", "nvidia_smi_log.gpu.default_applications_clocks.mem_clock": "877 MHz", "nvidia_smi_log.gpu.max_clocks.graphics_clock": "1530 MHz", "nvidia_smi_log.gpu.max_clocks.sm_clock": "1530 MHz", "nvidia_smi_log.gpu.max_clocks.mem_clock": "877 MHz", "nvidia_smi_log.gpu.max_clocks.video_clock": "1372 MHz", "nvidia_smi_log.gpu.max_customer_boost_clocks.graphics_clock": "1530 MHz", "nvidia_smi_log.gpu.processes.process_info.0.process_name": "python", "nvidia_smi_log.gpu.processes.process_info.1.process_name": "tritonserver", "hostname": "ip-10-100-8-98", "timestamp": 1615542360.9566503}
```

Pipeline example to read data from a file, run inference using an `abp-nvsmi-xgb` model, and write inference results to the specified output file.

```bash
helm install --set ngc.apiKey="$API_KEY" \
    --set sdk.args="morpheus --log_level=DEBUG run \
        --edge_buffer_size=4 \
        --pipeline_batch_size=1024 \
        --model_max_batch_size=64 \
        pipeline-fil --columns_file=data/columns_fil.txt \
          from-file --filename=./examples/data/nvsmi.jsonlines \
          monitor --description 'FromFile Rate' --smoothing=0.001 \
          deserialize \
          preprocess \
          monitor --description='Preprocessing Rate' \
          inf-triton --model_name=abp-nvsmi-xgb --server_url=ai-engine:8000 --force_convert_inputs=True \
          monitor --description='Inference Rate' --smoothing=0.001 --unit inf \
          add-class \
          serialize --exclude '^nvidia_smi_log' --exclude '^ts_' \
          to-file --filename=/common/data/<YOUR_OUTPUT_DIR>/abp-nvsmi-xgb-output.jsonlines --overwrite" \
    --namespace $NAMESPACE \
    <YOUR_RELEASE_NAME> \
    morpheus-sdk-client
```

Pipeline example to read messages from an input Kafka topic, run inference using an `abp-nvsmi-xgb` model, and write the results of the inference to an output Kafka topic:

```bash
helm install --set ngc.apiKey="$API_KEY" \
    --set sdk.args="morpheus --log_level=DEBUG run \
        --pipeline_batch_size=1024 \
        --model_max_batch_size=64 \
        pipeline-fil --columns_file=data/columns_fil.txt \
          from-kafka --input_topic <YOUR_INPUT_KAFKA_TOPIC> --bootstrap_servers broker:9092 \
          monitor --description 'FromKafka Rate' --smoothing=0.001 \
          deserialize \
          preprocess \
          monitor --description='Preprocessing Rate' \
          inf-triton --model_name=abp-nvsmi-xgb --server_url=ai-engine:8000 --force_convert_inputs=True \
          monitor --description='Inference Rate' --smoothing=0.001 --unit inf \
          add-class \
          serialize --exclude '^nvidia_smi_log' \ --exclude '^ts_' \
          to-kafka --output_topic <YOUR_OUTPUT_KAFKA_TOPIC> --bootstrap_servers broker:9092" \
    --namespace $NAMESPACE \
    <YOUR_RELEASE_NAME> \
    morpheus-sdk-client
```

Make sure you create input and output Kafka topics before you start the pipeline. After the pipeline has been started, load the individual corresponding data files from the downloaded sample into the selected input topic using the command below:

```bash
kubectl -n $NAMESPACE exec -it deploy/broker -c broker -- kafka-console-producer.sh \
       --broker-list broker:9092 \
       --topic <YOUR_INPUT_KAFKA_TOPIC> < \
       <YOUR_INPUT_DATA_FILE_PATH_EXAMPLE: ${HOME}/examples/data/nvsmi.jsonlines>
```

> **Note**: This should be used for development purposes only via this developer kit. Loading from the file into Kafka should not be used in production deployments of Morpheus.

### Verify Running Pipeline
Once you've deployed the SDK client to run a pipeline, you can check the status of the pod using the following command:

```bash
kubectl -n $NAMESPACE get pods sdk-cli-<RELEASE_NAME>
NAME                       READY   STATUS    RESTARTS   AGE
sdk-cli-6c9575f648-gfdd2   1/1     Running   0          3m23s
```

Then check that the pipeline is running successfully using the following command:

```bash
kubectl -n $NAMESPACE logs sdk-cli-<RELEASE_NAME>
```

Output:

```console
Configuring Pipeline via CLI
Starting pipeline via CLI... Ctrl+C to Quit
Preprocessing rate: 7051messages [00:09, 4372.75messages/s]
Inference rate: 7051messages [00:04, 4639.40messages/s]
```

## Prerequisites and Installation for AWS

### Prerequisites
1.  AWS account with the ability to create/modify EC2 instances
2.  AWS EC2 G4 instance with T4 or V100 GPU, at least 64GB RAM, 8 cores CPU, and 100 GB storage.

### Install Cloud Native Core Stack for AWS
On your AWS EC2 G4 instance, follow the instructions in the linked document to install [NVIDIA's Cloud Native Core Stack for AWS](https://github.com/NVIDIA/cloud-native-core).

## Prerequisites and Installation for Ubuntu

### Prerequisites
1.  NVIDIA-Certified System
2.  NVIDIA Volta GPU or newer (Compute Capability >= 7.0)
3.  Ubuntu 20.04 LTS or newer

## Installing Cloud Native Core Stack on NVIDIA Certified Systems
On your NVIDIA-Certified System, follow the instructions in the linked document to install [NVIDIA's Cloud Native Core Stack](https://github.com/NVIDIA/cloud-native-core).

## Kafka Topic Commands

List available Kafka topics.

```bash
kubectl -n $NAMESPACE exec deploy/broker -c broker -- kafka-topics.sh \
      --list  --zookeeper zookeeper:2181
```

Create a partitioned Kafka topic with a single replication factor.

```bash
kubectl -n $NAMESPACE exec deploy/broker -c broker -- kafka-topics.sh \
      --create \
      --bootstrap-server broker:9092 \
      --replication-factor 1 \
      --partitions 1 \
      --topic <YOUR_KAFKA_TOPIC>
```

Load data from a file to Kafka topic:

```bash
kubectl -n $NAMESPACE exec -it deploy/broker -c broker -- kafka-console-producer.sh \
       --broker-list broker:9092 \
       --topic <YOUR_KAFKA_TOPIC> < \
       <YOUR_INPUT_DATA_FILE>
```

> **Note**: This should be used for development purposes only via this developer kit. Loading from the file into Kafka should not be used in production deployments of Morpheus.


Consume messages from Kafka topic:

```bash
kubectl -n $NAMESPACE exec deploy/broker  -c broker -- kafka-console-consumer.sh \
       --bootstrap-server broker:9092 \
       --topic <YOUR_KAFKA_TOPIC> \
       --group <YOUR_CONSUMER_GROUP_ID>
```

Delete Kafka topic:

```bash
kubectl -n $NAMESPACE exec deploy/broker -c broker -- kafka-topics.sh \
      --delete --zookeeper zookeeper:2181 \
      --topic <YOUR_KAFKA_TOPIC>
```

## Additional Documentation
For more information on how to use the Morpheus Python API to customize and run your own optimized AI pipelines, Refer to below documentation.
- [Morpheus Developer Guides](./developer_guide/guides.md)
- [Morpheus Pipeline Examples](./examples.md)


## Troubleshooting
This section lists solutions to problems you might encounter with Morpheus or from its supporting components.

### Common Problems

- Models Unloaded After Reboot
  - When the pod is restarted, K8s will not automatically load the models. Since models are deployed to *ai-engine* in explicit mode using MLflow, we'd have to manually deploy them again using the [Model Deployment](#model-deployment) process.
- AI Engine CPU Only Mode
  - After a server restart, the ai-engine pod on k8s can start up before the GPU operator infrastructure is available, making it "think" there is no driver installed (that is, CPU -only mode).
- Improve Pipeline Message Processing Rate
  - Below settings need to be considered
    - Provide the workflow with the optimal number of threads (`—num threads`), as having more or fewer threads can have an impact on pipeline performance.
    - Consider adjusting `pipeline_batch_size` and `model_max_batch_size`
- Kafka Message Offset Commit Fail
  - Error Message
  ```console
  1649207839.253|COMMITFAIL|rdkafka#consumer-2| [thrd:main]: Offset commit (manual) failed for 1/1 partition(s) in join-state wait-unassign-call: Broker: Unknown member: topic[0]@112071(Broker: Unknown member)
  ```
  - Problem: If the standalone Kafka cluster is receiving significant message throughput from the producer, this error may happen.

  - Solution: Reinstall the Morpheus workflow and reduce the Kafka topic's message retention time and message producing rate.
