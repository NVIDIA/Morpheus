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

# Example Ransomware Detection Morpheus Pipeline for AppShield Data

Example of a Morpheus Pipeline using Triton Inference server.

## Setup Triton Inference Server

##### Pull Triton Inference Server Docker Image
Pull Docker image from NGC (https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver) suitable for your environment.

Example:

```bash
docker pull nvcr.io/nvidia/tritonserver:24.04-py3
```
##### Setup Env Variable
```bash
export MORPHEUS_ROOT=$(pwd)
```

##### Start Triton Inference Server Container
From the Morpheus repo root directory, run the following to launch Triton and load the `ransomw-model-short-rf` model:
```bash
# Run Triton in explicit mode
docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 \
    -v $PWD/examples/ransomware_detection/models:/models/triton-model-repo nvcr.io/nvidia/tritonserver:24.04-py3 \
    tritonserver --model-repository=/models/triton-model-repo \
                 --exit-on-error=false \
                 --model-control-mode=explicit \
                 --load-model ransomw-model-short-rf
```

##### Verify Model Deployment
Once Triton server finishes starting up, it will display the status of all loaded models. Successful deployment of the model will show the following:

```
+----------------------------+---------+--------+
| Model                      | Version | Status |
+----------------------------+---------+--------+
| ransomw-model-short-rf     | 1       | READY  |
+----------------------------+---------+--------+
```

> **Note**: If this is not present in the output, check the Triton log for any error messages related to loading the model.

## Requirements
> **Note**: Make sure `dask` and `distributed` are installed in your Conda environment before running the ransomware detection pipeline. Run the installation command specified below if not.

```bash
mamba install 'dask>=2023.1.1' 'distributed>=2023.1.1'
```

## Run Ransomware Detection Pipeline
Run the following from the root of the Morpheus repo to start the ransomware detection pipeline:

```bash
python examples/ransomware_detection/run.py --server_url=localhost:8001 \
              --sliding_window=3 \
              --model_name=ransomw-model-short-rf \
              --input_glob=./examples/data/appshield/*/snapshot-*/*.json \
              --output_file=./ransomware_detection_output.jsonlines
```

Input features for a short model can be taken from every three snapshots sequence, such as (1, 2, 3), or (2, 3, 4). The sliding window represents the number of subsequent snapshots that need to be taken into consideration when generating the input for a model. Sliding window for the medium model is `5` and for the long model it is `10`.

The configuration options for this example can be queried with:

```bash
python run.py --help
```

```
Usage: run.py [OPTIONS]

Options:
  --debug BOOLEAN
  --use_cpp BOOLEAN
  --num_threads INTEGER RANGE     Number of internal pipeline threads to use
                                  [x>=1]
  --n_dask_workers INTEGER RANGE  Number of dask workers  [x>=2]
  --threads_per_dask_worker INTEGER RANGE
                                  Number of threads per each dask worker
                                  [x>=2]
  --model_max_batch_size INTEGER RANGE
                                  Max batch size to use for the model  [x>=1]
  --model_fea_length INTEGER RANGE
                                  Features length to use for the model  [x>=1]
  --features_file TEXT            File path for ransomware detection features
  --model_name TEXT               The name of the model that is deployed on
                                  Tritonserver
  --server_url TEXT               Tritonserver url  [required]
  --sliding_window INTEGER RANGE  Sliding window to be used for model input
                                  request  [x>=1]
  --input_glob TEXT               Input glob pattern to match files to read.
                                  For example,
                                  './input_dir/*/snapshot-*/*.json' would read
                                  all files with the 'json' extension in the
                                  directory 'input_dir'.  [required]
  --watch_directory BOOLEAN       The watch directory option instructs this
                                  stage to not close down once all files have
                                  been read. Instead it will read all files
                                  that match the 'input_glob' pattern, and
                                  then continue to watch the directory for
                                  additional files. Any new files that are
                                  added that match the glob will then be
                                  processed.
  --output_file TEXT              The path to the file where the inference
                                  output will be saved.
  --help                          Show this message and exit.
  ```
