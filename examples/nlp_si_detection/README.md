<!--
SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Sensitive Information Detection with Natural Language Processing (NLP) Example

This example illustrates how to use Morpheus to detect Sensitive Information (SI) in network packets automatically by utilizing a Natural Language Processing (NLP) neural network and Triton Inference Server.

## Background

The goal of this example is to identify potentially sensitive information in network packet data as quickly as possible to limit exposure and take corrective action. Sensitive information is a broad term but can be generalized to any data that should be guarded from unauthorized access. Credit card numbers, passwords, authorization keys, and emails are all examples of sensitive information.

In this example, we will be using Morpheus' provided NLP SI Detection model. This model is capable of detecting 10 different categories of sensitive information:

1. Addresses
2. Bank Account Numbers
3. Credit Card Numbers
4. Email Addresses
5. Government ID Numbers
6. Personal Names
7. Passwords
8. Phone Numbers
9. Secret Keys (a.k.a Private Keys)
10. User IDs

### The Dataset

The dataset that this workflow was designed to process is PCAP, or Packet Capture data, that is serialized into a JSON format. Several different applications are capable of capurting this type of network traffic. Each packet contains information about the source, destination, timestamp, and body of the packet, among other things. For example, below is a single packet that is from a HTTP POST request to cumulusnetworks.com:

```json
{
  "timestamp": 1616380971990,
  "host_ip": "10.188.40.56",
  "data_len": "309",
  "data": "POST /simpledatagen/ HTTP/1.1\r\nHost: echo.gtc1.netqdev.cumulusnetworks.com\r\nUser-Agent: python-requests/2.22.0\r\nAccept-Encoding: gzip, deflate\r\nAccept: */*\r\nConnection: keep-alive\r\nContent-Length: 73\r\nContent-Type: application/json\r\n\r\n",
  "src_mac": "04:3f:72:bf:af:74",
  "dest_mac": "b4:a9:fc:3c:46:f8",
  "protocol": "6",
  "src_ip": "10.20.16.248",
  "dest_ip": "10.244.0.59",
  "src_port": "50410",
  "dest_port": "80",
  "flags": "24"
}
```

In this example, we will be using a simulated PCAP dataset that is known to contain SI from each of the 10 categories the model was trained for. The dataset is located at `examples/data/pcap_dump.jsonlines`. The dataset is in the `.jsonlines` format which means each new line represents a new JSON object. In order to parse this data, it must be ingested, split by lines into individual JSON objects, and parsed. This will all be handled by Morpheus.

## Pipeline Architecture

The pipeline we will be using in this example is a simple feed-forward linear pipeline where the data from each stage flows on to the next. Simple linear pipelines with no custom stages, like this example, can be configured via the Morpheus CLI or using the Python library. In this example we will be using the Morpheus CLI.

Below is a visualization of the pipeline showing all of the stages and data types as it flows from one stage to the next.

![Pipeline](pipeline.png)


## Setup

This example utilizes the Triton Inference Server to perform inference. The neural network model is provided in the `models/sid-models` directory of the Morpheus repo.

### Launching Triton

From the Morpheus repo root directory, run the following to launch Triton and load the `sid-minibert` model:

```bash
docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 -v $PWD/models:/models nvcr.io/nvidia/tritonserver:24.04-py3 tritonserver --model-repository=/models/triton-model-repo --exit-on-error=false --model-control-mode=explicit --load-model sid-minibert-onnx
```

Where `24.04-py3` can be replaced with the current year and month of the Triton version to use. For example, to use May 2021, specify `nvcr.io/nvidia/tritonserver:21.05-py3`. Ensure that the version of TensorRT that is used in Triton matches the version of TensorRT elsewhere (refer to [NGC Deep Learning Frameworks Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)).

This will launch Triton and only load the `sid-minibert-onnx` model. This model has been configured with a max batch size of 32, and to use dynamic batching for increased performance.

Once Triton has loaded the model, the following should be displayed:

```
+-------------------+---------+--------+
| Model             | Version | Status |
+-------------------+---------+--------+
| sid-minibert-onnx | 1       | READY  |
+-------------------+---------+--------+
```
> **Note**: If this is not present in the output, check the Triton log for any error messages related to loading the model.

## Running the Pipeline

With the Morpheus CLI, an entire pipeline can be configured and run without writing any code. Using the `morpheus run pipeline-nlp` command, we can build the pipeline by specifying each stage's name and configuration right on the command line. The output of each stage will become the input for the next.

The following command line is the entire command to build and launch the pipeline. Each new line represents a new stage. The comment above each stage gives information about why the stage was added and configured this way.

From the Morpheus repo root directory, run:
```bash
# Launch Morpheus printing debug messages
morpheus --log_level=DEBUG \
   `# Run a pipeline with a model batch size of 32 (Must match Triton config)` \
   run --pipeline_batch_size=1024 --model_max_batch_size=32 \
   `# Specify a NLP pipeline with 256 sequence length (Must match Triton config)` \
   pipeline-nlp --model_seq_length=256 \
   `# 1st Stage: Read from file` \
   from-file --filename=examples/data/pcap_dump.jsonlines \
   `# 2nd Stage: Deserialize from JSON strings to objects` \
   deserialize \
   `# 3rd Stage: Preprocessing converts the input data into BERT tokens` \
   preprocess --vocab_hash_file=data/bert-base-uncased-hash.txt --do_lower_case=True --truncation=True \
   `# 4th Stage: Send messages to Triton for inference. Specify the model loaded in Setup` \
   inf-triton --model_name=sid-minibert-onnx --server_url=localhost:8000 --force_convert_inputs=True \
   `# 5th Stage: Monitor stage prints throughput information to the console` \
   monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
   `# 6th Stage: Add results from inference to the messages` \
   add-class \
   `# 7th Stage: Filtering removes any messages that did not detect SI` \
   filter --filter_source=TENSOR \
   `# 8th Stage: Convert from objects back into strings` \
   serialize --exclude '^_ts_' \
   `# 9th Stage: Write out the JSON lines to the detections.jsonlines file` \
   to-file --filename=detections.jsonlines --overwrite
```

If successful, the following should be displayed:

```bash
Configuring Pipeline via CLI
Loaded labels file. Current labels: [['address', 'bank_acct', 'credit_card', 'email', 'govt_id', 'name', 'password', 'phone_num', 'secret_keys', 'user']]
Starting pipeline via CLI... Ctrl+C to Quit
Config:
{
  "ae": null,
  "class_labels": [
    "address",
    "bank_acct",
    "credit_card",
    "email",
    "govt_id",
    "name",
    "password",
    "phone_num",
    "secret_keys",
    "user"
  ],
  "debug": false,
  "edge_buffer_size": 128,
  "feature_length": 256,
  "fil": null,
  "log_config_file": null,
  "log_level": 10,
  "mode": "NLP",
  "model_max_batch_size": 32,
  "num_threads": 8,
  "pipeline_batch_size": 1024
}
CPP Enabled: True
====Registering Pipeline====
====Registering Pipeline Complete!====
====Starting Pipeline====
====Pipeline Started====
====Building Pipeline====
Added source: <from-file-0; FileSourceStage(filename=examples/data/pcap_dump.jsonlines, iterative=False, file_type=FileTypes.Auto, repeat=1, filter_null=True)>
  └─> morpheus.MessageMeta
Added stage: <deserialize-1; DeserializeStage()>
  └─ morpheus.MessageMeta -> morpheus.MultiMessage
Added stage: <preprocess-nlp-2; PreprocessNLPStage(vocab_hash_file=/opt/conda/envs/morpheus/lib/python3.8/site-packages/morpheus/data/bert-base-uncased-hash.txt, truncation=True, do_lower_case=True, add_special_tokens=False, stride=-1)>
  └─ morpheus.MultiMessage -> morpheus.MultiInferenceNLPMessage
Added stage: <inference-3; TritonInferenceStage(model_name=sid-minibert-onnx, server_url=localhost:8000, force_convert_inputs=True, use_shared_memory=False)>
  └─ morpheus.MultiInferenceNLPMessage -> morpheus.MultiResponseMessage
Added stage: <monitor-4; MonitorStage(description=Inference Rate, smoothing=0.001, unit=inf, delayed_start=False, determine_count_fn=None)>
  └─ morpheus.MultiResponseMessage -> morpheus.MultiResponseMessage
Added stage: <add-class-5; AddClassificationsStage(threshold=0.5, labels=[], prefix=)>
  └─ morpheus.MultiResponseMessage -> morpheus.MultiResponseMessage
Added stage: <serialize-6; SerializeStage(include=[], exclude=['^_ts_'], fixed_columns=True)>
  └─ morpheus.MultiResponseMessage -> morpheus.MessageMeta
Added stage: <to-file-7; WriteToFileStage(filename=detections.jsonlines, overwrite=True, file_type=FileTypes.Auto)>
  └─ morpheus.MessageMeta -> morpheus.MessageMeta
====Building Pipeline Complete!====
Starting! Time: 1656352480.541071
Inference Rate[Complete]: 93085inf [00:07, 12673.63inf/s]
====Pipeline Complete====

```

The output file `detections.jsonlines` will contain the original PCAP messages with the following additional fields added:
* address
* bank_acct
* credit_card
* email
* govt_id
* name
* password
* phone_num
* secret_keys
* user

The value for these fields will be a `1` indicating a detection or a `0` indicating no detection. An example row with a detection is:
```json
{
  "timestamp": 1616381019580,
  "host_ip": "10.188.40.56",
  "data_len": "129",
  "data": "\"{\\\"X-Postmark-Server-Token\\\": \\\"76904958 O7FWqd9p TzIBfSYk\\\"}\"",
  "src_mac": "04:3f:72:bf:af:74",
  "dest_mac": "b4:a9:fc:3c:46:f8",
  "protocol": "6",
  "src_ip": "10.20.16.248",
  "dest_ip": "10.244.0.60",
  "src_port": "51374",
  "dest_port": "80",
  "flags": "24",
  "is_pii": false,
  "address": 0,
  "bank_acct": 0,
  "credit_card": 0,
  "email": 0,
  "govt_id": 0,
  "name": 0,
  "password": 0,
  "phone_num": 0,
  "secret_keys": 1,
  "user": 0
}
```
