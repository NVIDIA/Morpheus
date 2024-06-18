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

# Anomalous Behavior Profiling with Forest Inference Library (FIL) Example

This example illustrates how to use Morpheus to automatically detect abnormal behavior in NVIDIA SMI logs by utilizing a Forest Inference Library (FIL) model and Triton Inference Server. The particular behavior we will be searching for is cryptocurrency mining.

## Background

The goal of this example is to identify whether or not a monitored NVIDIA GPU is actively mining for cryptocurrencies and take corrective action if detected. Cryptocurrency mining can be a large resource drain on GPU clusters and detecting mining can be difficult since mining workloads appear similar to other valid workloads.

In this example, we will be using Morpheus' provided ABP NVSMI Detection model. This model is capable of detecting the signature of cryptocurrency mining from the output of `nvidia-smi` logs. For each timestamp that `nvidia-smi` log data is available, the model will output a single probability indicating whether mining was detected or not.

### The Dataset

The dataset that this workflow was designed to process contains NVIDIA GPU metrics at regular time intervals and is extracted by a NetQ agent and serialized into JSON. Each line in the dataset contains much of the same information that is returned by the `nvidia-smi` utility. We won't examine at a full message directly since each line contains 176 different columns, but it's possible to get an idea of how the dataset was generated using the `nvidia-smi dmon` command. If you run this yourself, the output similar to the following:

```bash
$ nvidia-smi dmon
# gpu   pwr gtemp mtemp    sm   mem   enc   dec  mclk  pclk
# Idx     W     C     C     %     %     %     %   MHz   MHz
    0    70    48     -     5     1     0     0  7000  1350
    0    68    48     -    11     1     0     0  7000  1350
    0    69    48     -     3     1     0     0  7000  1350
    0   270    53     -    10     1     0     0  7000  1875
    0   274    55     -    75    46     0     0  7000  1740
    0   278    55     -    86    56     0     0  7000  1755
    0   279    56     -    99    63     0     0  7000  1755
    0   277    57     -    86    55     0     0  7000  1755
    0   281    57     -    85    54     0     0  7000  1740
```

Each line in the output represents the GPU metrics at a single point in time. As the tool progresses the GPU begins to be utilized and the SM% and Mem% values increase as memory is loaded into the GPU and computations are performed. The model we will be using can ingest this information and determine whether or not the GPU is mining cryptocurrencies without needing additional information from the host machine.

In this example we will be using the `examples/data/nvsmi.jsonlines` dataset that is known to contain mining behavior profiles. The dataset is in the `.jsonlines` format which means each new line represents a new JSON object. In order to parse this data, it must be ingested, split by lines into individual JSON objects, and parsed into cuDF dataframes. This will all be handled by Morpheus.

#### Generating your own dataset

This example can be easily applied to datasets generated from your own NVIDIA GPU devices. If NetQ is not deployed in your environment, the `nvsmi_data_extract.py` script is provided which uses [pyNVML](https://pypi.org/project/nvidia-ml-py/) and [pandas](https://pandas.pydata.org/) to generate data similar to NetQ. `pyNVML` contains the Python bindings for NVIDIA Management Library (NVML), the same library used by `nvidia-smi`.

`pyNVML` and `pandas` come already installed on the Morpheus release and development Docker images. Otherwise, they will need to be installed before running the script.

Run the following to start generating your dataset:
```
python nvsmi_data_extract.py
```
This will write a new entry to an output file named `nvsmi.jsonlines` once per second until you press Ctrl+C to exit.

## Pipeline Architecture

The pipeline we will be using in this example is a simple feed-forward linear pipeline where the data from each stage flows on to the next. Simple linear pipelines with no custom stages, like this example, can be configured via the Morpheus CLI or using the Python library. In this example we will be using the Morpheus CLI.

Below is a visualization of the pipeline showing all of the stages and data types as it flows from one stage to the next.

![Pipeline](pipeline.png)


## Setup

This example utilizes the Triton Inference Server to perform inference.

### Launching Triton

Pull the Docker image for Triton:
```bash
docker pull nvcr.io/nvidia/tritonserver:24.04-py3
```

From the Morpheus repo root directory, run the following to launch Triton and load the `abp-nvsmi-xgb` XGBoost model:
```bash
docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 -v $PWD/models:/models nvcr.io/nvidia/tritonserver:24.04-py3 tritonserver --model-repository=/models/triton-model-repo --exit-on-error=false --model-control-mode=explicit --load-model abp-nvsmi-xgb
```

This will launch Triton and only load the `abp-nvsmi-xgb` model. This model has been configured with a max batch size of 32768, and to use dynamic batching for increased performance.

Once Triton has loaded the model, the following will be displayed:

```
+-------------------+---------+--------+
| Model             | Version | Status |
+-------------------+---------+--------+
| abp-nvsmi-xgb     | 1       | READY  |
+-------------------+---------+--------+
```

> **Note**: If this is not present in the output, check the Triton log for any error messages related to loading the model.

## Running the Pipeline

With the Morpheus CLI, an entire pipeline can be configured and run without writing any code. Using the `morpheus run pipeline-fil` command, we can build the pipeline by specifying each stage's name and configuration right on the command line. The output of each stage will become the input for the next.

The following command line is the entire command to build and launch the pipeline. Each new line represents a new stage. The comment above each stage gives information about why the stage was added and configured this way (you can copy/paste the entire command with comments).

From the  Morpheus repo root directory, run:
```bash
# Launch Morpheus printing debug messages
morpheus --log_level=DEBUG \
   `# Run a pipeline with 8 threads and a model batch size of 1024 (Must be equal or less than Triton config)` \
   run --num_threads=8 --pipeline_batch_size=1024 --model_max_batch_size=1024 \
   `# Specify a NLP pipeline with 256 sequence length (Must match Triton config)` \
   pipeline-fil --columns_file=data/columns_fil.txt \
   `# 1st Stage: Read from file` \
   from-file --filename=examples/data/nvsmi.jsonlines \
   `# 2nd Stage: Deserialize from JSON strings to objects` \
   deserialize \
   `# 3rd Stage: Preprocessing converts the input data into BERT tokens` \
   preprocess \
   `# 4th Stage: Send messages to Triton for inference. Specify the model loaded in Setup` \
   inf-triton --model_name=abp-nvsmi-xgb --server_url=localhost:8000 \
   `# 5th Stage: Monitor stage prints throughput information to the console` \
   monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
   `# 6th Stage: Add results from inference to the messages` \
   add-class \
   `# 7th Stage: Convert from objects back into strings. Ignore verbose input data` \
   serialize --include 'mining' \
   `# 8th Stage: Write out the JSON lines to the detections.jsonlines file` \
   to-file --filename=detections.jsonlines --overwrite
```

If successful, the following should be displayed:
```bash
Configuring Pipeline via CLI
Loaded columns. Current columns: [['nvidia_smi_log.gpu.fb_memory_usage.used', 'nvidia_smi_log.gpu.fb_memory_usage.free', 'nvidia_smi_log.gpu.utilization.gpu_util', 'nvidia_smi_log.gpu.utilization.memory_util', 'nvidia_smi_log.gpu.temperature.gpu_temp', 'nvidia_smi_log.gpu.temperature.gpu_temp_max_threshold', 'nvidia_smi_log.gpu.temperature.gpu_temp_slow_threshold', 'nvidia_smi_log.gpu.power_readings.power_draw', 'nvidia_smi_log.gpu.clocks.graphics_clock', 'nvidia_smi_log.gpu.clocks.sm_clock', 'nvidia_smi_log.gpu.clocks.mem_clock', 'nvidia_smi_log.gpu.applications_clocks.graphics_clock', 'nvidia_smi_log.gpu.applications_clocks.mem_clock', 'nvidia_smi_log.gpu.default_applications_clocks.graphics_clock', 'nvidia_smi_log.gpu.default_applications_clocks.mem_clock', 'nvidia_smi_log.gpu.max_clocks.graphics_clock', 'nvidia_smi_log.gpu.max_clocks.sm_clock', 'nvidia_smi_log.gpu.max_clocks.mem_clock']]
Starting pipeline via CLI... Ctrl+C to Quit
Config:
{
  "ae": null,
  "class_labels": [
    "mining"
  ],
  "debug": false,
  "edge_buffer_size": 128,
  "feature_length": 18,
  "fil": {
    "feature_columns": [
      "nvidia_smi_log.gpu.pci.tx_util",
      "nvidia_smi_log.gpu.pci.rx_util",
      "nvidia_smi_log.gpu.fb_memory_usage.used",
      "nvidia_smi_log.gpu.fb_memory_usage.free",
      "nvidia_smi_log.gpu.bar1_memory_usage.total",
      "nvidia_smi_log.gpu.bar1_memory_usage.used",
      "nvidia_smi_log.gpu.bar1_memory_usage.free",
      "nvidia_smi_log.gpu.utilization.gpu_util",
      "nvidia_smi_log.gpu.utilization.memory_util",
      "nvidia_smi_log.gpu.temperature.gpu_temp",
      "nvidia_smi_log.gpu.temperature.gpu_temp_max_threshold",
      "nvidia_smi_log.gpu.temperature.gpu_temp_slow_threshold",
      "nvidia_smi_log.gpu.temperature.gpu_temp_max_gpu_threshold",
      "nvidia_smi_log.gpu.temperature.memory_temp",
      "nvidia_smi_log.gpu.temperature.gpu_temp_max_mem_threshold",
      "nvidia_smi_log.gpu.power_readings.power_draw",
      "nvidia_smi_log.gpu.clocks.graphics_clock",
      "nvidia_smi_log.gpu.clocks.sm_clock",
      "nvidia_smi_log.gpu.clocks.mem_clock",
      "nvidia_smi_log.gpu.clocks.video_clock",
      "nvidia_smi_log.gpu.applications_clocks.graphics_clock",
      "nvidia_smi_log.gpu.applications_clocks.mem_clock",
      "nvidia_smi_log.gpu.default_applications_clocks.graphics_clock",
      "nvidia_smi_log.gpu.default_applications_clocks.mem_clock",
      "nvidia_smi_log.gpu.max_clocks.graphics_clock",
      "nvidia_smi_log.gpu.max_clocks.sm_clock",
      "nvidia_smi_log.gpu.max_clocks.mem_clock",
      "nvidia_smi_log.gpu.max_clocks.video_clock",
      "nvidia_smi_log.gpu.max_customer_boost_clocks.graphics_clock"
    ]
  },
  "log_config_file": null,
  "log_level": 10,
  "mode": "FIL",
  "model_max_batch_size": 1024,
  "num_threads": 8,
  "pipeline_batch_size": 1024
}
CPP Enabled: True
====Registering Pipeline====
====Registering Pipeline Complete!====
====Starting Pipeline====
====Pipeline Started====
====Building Pipeline====
Added source: <from-file-0; FileSourceStage(filename=examples/data/nvsmi.jsonlines, iterative=False, file_type=FileTypes.Auto, repeat=1, filter_null=True)>
  └─> morpheus.MessageMeta
Added stage: <deserialize-1; DeserializeStage()>
  └─ morpheus.MessageMeta -> morpheus.MultiMessage
Added stage: <preprocess-fil-2; PreprocessFILStage()>
  └─ morpheus.MultiMessage -> morpheus.MultiInferenceFILMessage
Added stage: <inference-3; TritonInferenceStage(model_name=abp-nvsmi-xgb, server_url=localhost:8000, force_convert_inputs=False, use_shared_memory=False)>
  └─ morpheus.MultiInferenceFILMessage -> morpheus.MultiResponseMessage
Added stage: <monitor-4; MonitorStage(description=Inference Rate, smoothing=0.001, unit=inf, delayed_start=False, determine_count_fn=None)>
  └─ morpheus.MultiResponseMessage -> morpheus.MultiResponseMessage
Added stage: <add-class-5; AddClassificationsStage(threshold=0.5, labels=[], prefix=)>
  └─ morpheus.MultiResponseMessage -> morpheus.MultiResponseMessage
Added stage: <serialize-6; SerializeStage(include=['mining'], exclude=['^ID$', '^_ts_'], fixed_columns=True)>
  └─ morpheus.MultiResponseMessage -> morpheus.MessageMeta
Added stage: <to-file-7; WriteToFileStage(filename=detections.jsonlines, overwrite=True, file_type=FileTypes.Auto)>
  └─ morpheus.MessageMeta -> morpheus.MessageMeta
====Building Pipeline Complete!====
Starting! Time: 1656353254.9919598
Inference Rate[Complete]: 1242inf [00:00, 1863.04inf/s]
====Pipeline Complete====
```

The output file `detections.jsonlines` will contain a single boolean value for each input line. At some point the values will switch from `0` to `1`:

```
...
{"mining": 0}
{"mining": 0}
{"mining": 0}
{"mining": 0}
{"mining": 1}
{"mining": 1}
{"mining": 1}
{"mining": 1}
{"mining": 1}
{"mining": 1}
{"mining": 1}
{"mining": 1}
...
```

 We have stripped out the input data to make the detections easier to identify. Omitting the argument `--include 'mining'` would show the input data in the detections file.
