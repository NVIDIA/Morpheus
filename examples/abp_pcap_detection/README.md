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

# ABP Detection Example Using Morpheus


## Setup
To run this example, an instance of Triton Inference Server and a sample dataset is required. The following steps will outline how to build and run Triton with the provided FIL model.

### Triton Inference Server
```bash
docker pull nvcr.io/nvidia/tritonserver:24.04-py3
```

##### Deploy Triton Inference Server
From the root of the Morpheus repo, run the following to launch Triton and load the `abp-pcap-xgb` model:
```bash
docker run --rm --gpus=all -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $PWD/examples/abp_pcap_detection/abp-pcap-xgb:/models/abp-pcap-xgb --name tritonserver nvcr.io/nvidia/tritonserver:24.04-py3 tritonserver --model-repository=/models --exit-on-error=false
```

##### Verify Model Deployment
Once Triton server finishes starting up, it will display the status of all loaded models. Successful deployment of the model will show the following:

```bash
+-----------------------------+---------+--------+
| Model                       | Version | Status |
+-----------------------------+---------+--------+
| abp-pcap-xgb                | 1       | READY  |
+-----------------------------+---------+--------+
```

## ABP Detection Pipeline
Use Morpheus to run the Anomalous Behavior Profiling Detection Pipeline with the pcap data. A pipeline has been configured in `run.py` with several command line options:

From the root of the Morpheus repo, run:
```bash
python examples/abp_pcap_detection/run.py --help
```

Output:
```
Usage: run.py [OPTIONS]

Options:
  --num_threads INTEGER RANGE     Number of internal pipeline threads to use.
                                  [x>=1]
  --pipeline_batch_size INTEGER RANGE
                                  Internal batch size for the pipeline. Can be
                                  much larger than the model batch size. Also
                                  used for Kafka consumers.  [x>=1]
  --model_max_batch_size INTEGER RANGE
                                  Max batch size to use for the model.  [x>=1]
  --input_file PATH               Input filepath.  [required]
  --output_file TEXT              The path to the file where the inference
                                  output will be saved.
  --model_fea_length INTEGER RANGE
                                  Features length to use for the model.
                                  [x>=1]
  --model_name TEXT               The name of the model that is deployed on
                                  Tritonserver.
  --iterative                     Iterative mode will emit dataframes one at a
                                  time. Otherwise a list of dataframes is
                                  emitted. Iterative mode is good for
                                  interleaving source stages.
  --server_url TEXT               Tritonserver url.  [required]
  --file_type [auto|csv|json]     Indicates what type of file to read.
                                  Specifying 'auto' will determine the file
                                  type from the extension.
  --help                          Show this message and exit.
```

To launch the configured Morpheus pipeline with the sample data that is provided in `examples/data`, run the following:

```bash
python examples/abp_pcap_detection/run.py
```
Note: Both Morpheus and Triton Inference Server containers must have access to the same GPUs in order for this example to work.

The pipeline will process the input `abp_pcap_dump.jsonlines` sample data and write it to `pcap_out.jsonlines`.

### CLI Example
The above example is illustrative of using the Python API to build a custom Morpheus Pipeline.
Alternately, the Morpheus command line could have been used to accomplish the same goal by registering the `abp_pcap_preprocessing.py` module as a plugin.

From the root of the Morpheus repo, run:
```bash
morpheus --log_level INFO --plugin "examples/abp_pcap_detection/abp_pcap_preprocessing.py" \
    run --pipeline_batch_size 100000 --model_max_batch_size 100000 \
    pipeline-fil --model_fea_length 13 --label=probs \
    from-file --filename examples/data/abp_pcap_dump.jsonlines --filter_null False \
    deserialize \
    pcap-preprocess \
    monitor --description "Preprocessing rate" \
    inf-triton --model_name "abp-pcap-xgb" --server_url "localhost:8000" \
    monitor --description "Inference rate" --unit inf \
    add-class --label=probs \
    monitor --description "Add classification rate" --unit "add-class" \
    serialize \
    monitor --description "Serialize rate" --unit ser \
    to-file --filename "pcap_out.jsonlines" --overwrite \
    monitor --description "Write to file rate" --unit "to-file"
```
