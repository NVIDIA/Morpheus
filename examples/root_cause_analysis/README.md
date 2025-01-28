<!--
SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Root Cause Analysis Acceleration & Predictive Maintenance Example

These examples illustrate how to use Morpheus to build a binary sequence classification pipelines to perform root cause analysis on DGX kernel logs.

## Supported Environments
| Environment | Supported | Notes |
|-------------|-----------|-------|
| Conda | ✔ | |
| Morpheus Docker Container | ✔ | Requires launching Triton on the host |
| Morpheus Release Container | ✔ | Requires launching Triton on the host |
| Dev Container | ✔ | Requires using the `dev-triton-start` script and replacing `--server_url=localhost:8000` with `--server_url=triton:8000` |

## Background

Like any other Linux based machine, DGX systems generate a vast amount of logs. Analysts spend hours trying to identify the root causes of each failure. There could be infinitely many types of root causes of the failures. Some patterns might help to narrow it down; however, regular expressions can only help to identify previously known patterns. Moreover, this creates another manual task of maintaining a search script.

In this example, we demonstrate how using Morpheus can accelerate the analysis of the enormous amount of logs using machine learning. Another benefit of analyzing in a probabilistic way is that we can pin down previously undetected root causes. To achieve this, we will fine-tune a pre-trained BERT[^1] model with a classification layer using HuggingFace library.

Once the model is capable of identifying even the new root causes, it can also be deployed as a process running in the machines to predict failures before they happen.

[^1]: BERT stands for Bidirectional Encoder Representations from Transformers. The paper can be found [here](https://arxiv.org/pdf/1810.04805.pdf).

### The Dataset

The dataset comprises kern.log files from multiple DGX systems. Each line inside has been labelled as either 0 for ordinary or 1 for root cause by a script that uses some known patterns. We will be especially interested in lines that are marked as ordinary in the test set but predicted as a root cause as they may be new types of root causes of failures.

## Pipeline Architecture

The pipeline we will be using in this example is a simple feed-forward linear pipeline where the data from each stage flows on to the next. Morpheus pipelines can be configured via the Morpheus CLI or using the Python library. In this example we will be using the Morpheus CLI.

## Setup

This example utilizes the Triton Inference Server to perform inference. The binary sequence classification neural network model is provided in the `models/root-cause-models` directory of the Morpheus repo.

### Launching Triton

From the Morpheus repo root directory, run the following to launch Triton and load the `root-cause-binary-onnx` model:

```bash
docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 nvcr.io/nvidia/morpheus/morpheus-tritonserver-models:25.02 tritonserver --model-repository=/models/triton-model-repo --exit-on-error=false --model-control-mode=explicit --load-model root-cause-binary-onnx
```

This will launch Triton and only load the model required by our example pipeline. The model has been configured with a max batch size of 32, and to use dynamic batching for increased performance.

Once Triton has loaded the model, the following should be displayed:

```
+----------------------------+-----+--------+
| Model                  | Version | Status |
+------------------------+---------+--------+
| root-cause-binary-onnx | 1       | READY  |
+------------------------+---------+--------+

```
> **Note**: If this is not present in the output, check the Triton log for any error messages related to loading the model.

#### ONNX->TensorRT Model Conversion

To achieve optimized inference performance, Triton Inference Server provides an option to convert our ONNX model to TensorRT. Simply add the following to the end of your `config.pbtxt`:
```
dynamic_batching {
  preferred_batch_size: [ 1, 4, 8, 16, 32 ]
  max_queue_delay_microseconds: 50000
}

optimization { execution_accelerators {
  gpu_execution_accelerator : [ {
    name : "tensorrt"
    parameters { key: "precision_mode" value: "FP16" }
    parameters { key: "max_workspace_size_bytes" value: "1073741824" }
    }]
}}
```
This will trigger the ONNX->TensorRT model conversion upon first use of the model. Therefore, expect only the first pipeline run to take several minutes to allow for this conversion. You should notice substantial inference speedups compared to ONNX in subsequent pipeline runs.

More information about TensorRT can be found [here](https://developer.nvidia.com/tensorrt).

## Running the Pipeline

With the Morpheus CLI, an entire pipeline can be configured and run without writing any code. Using the `morpheus run pipeline-nlp` command, we can build the pipeline by specifying each stage's name and configuration right on the command line. The output of each stage will become the input for the next.

### Root Cause Analysis using Binary Sequence Classification

The following command line is the entire command to build and launch a root cause analysis pipeline which uses binary sequence classification. Each new line represents a new stage. The comment above each stage gives information about why the stage was added and configured this way.

From the Morpheus repo root directory, run:

```bash
export MORPHEUS_ROOT=$(pwd)
morpheus --log_level=DEBUG \
`# Run a pipeline with 8 threads and a model batch size of 32 (Must match Triton config)` \
run --num_threads=8 --edge_buffer_size=4 --pipeline_batch_size=1024 --model_max_batch_size=32 \
`# Specify a NLP pipeline with 128 sequence length (Must match Triton config)` \
pipeline-nlp --model_seq_length=128 --label=not_root_cause --label=is_root_cause \
`# 1st Stage: Read from file` \
from-file --filename=${MORPHEUS_ROOT}/examples/data/root-cause-validation-data-input.jsonlines \
`# 2nd Stage: Deserialize batch DataFrame into ControlMessages` \
deserialize \
`# 3rd Stage: Preprocessing converts the input data into BERT tokens` \
preprocess --column=log --vocab_hash_file=./data/bert-base-uncased-hash.txt --truncation=True --do_lower_case=True --add_special_tokens=False \
`# 4th Stage: Send messages to Triton for inference. Specify the binary model loaded in Setup` \
inf-triton --model_name=root-cause-binary-onnx --server_url=localhost:8000 --force_convert_inputs=True \
`# 5th Stage: Monitor stage prints throughput information to the console` \
monitor --description='Inference rate' --smoothing=0.001 --unit inf \
`# 6th Stage: Add scores from inference to the messages` \
add-scores --label=is_root_cause \
`# 7th Stage: Convert from objects back into strings` \
serialize --exclude '^ts_' \
`# 8th Stage: Write results out to CSV file` \
to-file --filename=.tmp/output/root-cause-binary-output.jsonlines --overwrite
```

If successful, the following should be displayed:

```bash
Configuring Pipeline via CLI
Loaded labels file. Current labels: [['not_root_cause', 'is_root_cause']]
Parameter, 'vocab_hash_file', with relative path, './data/bert-base-uncased-hash.txt', does not exist. Using package relative location: '/opt/conda/envs/morpheus/lib/python3.8/site-packages/morpheus/./data/bert-base-uncased-hash.txt'
Starting pipeline via CLI... Ctrl+C to Quit
Config:
{
  "ae": null,
  "class_labels": [
    "not_root_cause",
    "is_root_cause"
  ],
  "debug": false,
  "edge_buffer_size": 4,
  "feature_length": 128,
  "fil": null,
  "log_config_file": null,
  "log_level": 10,
  "mode": "NLP",
  "model_max_batch_size": 32,
  "num_threads": 8,
  "pipeline_batch_size": 1024,
  "plugins": []
}
CPP Enabled: True
====Registering Pipeline====
====Building Pipeline====
====Building Segment: linear_segment_0====
====Building Segment Complete!====
====Building Pipeline Complete!====
Starting! Time: 1668537665.9479523
====Registering Pipeline Complete!====
====Starting Pipeline====
====Pipeline Started====
Added source: <from-file-0; FileSourceStage(filename=/workspace/models/datasets/validation-data/root-cause-validation-data-input.jsonlines, iterative=False, file_type=FileTypes.Auto, repeat=1, filter_null=True)>
  └─> morpheus.MessageMeta
Added stage: <deserialize-1; DeserializeStage()>
  └─ morpheus.MessageMeta -> morpheus.ControlMessage
Added stage: <preprocess-nlp-2; PreprocessNLPStage(vocab_hash_file=/opt/conda/envs/morpheus/lib/python3.8/site-packages/morpheus/data/bert-base-uncased-hash.txt, truncation=True, do_lower_case=True, add_special_tokens=False, stride=-1, column=log)>
  └─ morpheus.ControlMessage -> morpheus.ControlMessage
Added stage: <inference-3; TritonInferenceStage(model_name=root-cause-binary-onnx, server_url=localhost:8001, force_convert_inputs=True, use_shared_memory=False)>
  └─ morpheus.ControlMessage -> morpheus.ControlMessage
Added stage: <monitor-4; MonitorStage(description=Inference rate, smoothing=0.001, unit=inf, delayed_start=False, determine_count_fn=None)>
  └─ morpheus.ControlMessage -> morpheus.ControlMessage
Added stage: <add-scores-5; AddScoresStage(labels=('is_root_cause',), prefix=)>
  └─ morpheus.ControlMessagee -> morpheus.ControlMessage
Added stage: <serialize-6; SerializeStage(include=(), exclude=('^ts_',), fixed_columns=True)>
  └─ morpheus.ControlMessage -> morpheus.MessageMeta
Added stage: <to-file-7; WriteToFileStage(filename=.tmp/output/root-cause-binary-output.jsonlines, overwrite=True, file_type=FileTypes.Auto, include_index_col=True)>
  └─ morpheus.MessageMeta -> morpheus.MessageMeta
Inference rate[Complete]: 473 inf [00:01, 340.43 inf/s]
====Pipeline Complete====
```

The output file `.tmp/output/root-cause-binary-output.jsonlines` will contain the original kernel log messages with an additional field `is_root_cause`. The value of the new field will be the root cause probability.
