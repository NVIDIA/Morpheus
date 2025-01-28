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

# Triton Model Repository

This directory contains the necessary directory structure and configuration files in order to run the Morpheus Models in Triton Inference Server.

Each directory in the Triton Model repo contains a single model and configuration file. The model file is stored in a directory indicating the version number (by default this is `1`). The model file itself is a symlink to a specific model file elsewhere in the repo.

For example, the Triton model `sid-minibert-onnx` can be found in the `triton-model-repo` directory with the following layout:

```
triton-model-repo/
   sid-minibert-onnx/
      1/
         model.onnx -> ../../../sid-models/sid-bert-20211021.onnx
      config.pbtxt
```

## Symbolic Links

Sym links are used to minimize changes to the `config.pbtxt` files while still allowing for new models to be added at a future date. Without symlinks, each `config.pbtxt` would need to update the `default_model_filename:` option each time the model was changed.

The downside of using symlinks is that the entire Morpheus model repo must be volume mounted when launching Triton. Refer to the next section for information on how to correctly mount this repo, and select which models should be loaded.

## Models Container
The models in this directory are available in a pre-built container image containing Triton Inference Server, along with the models themselves. The container image is available on NGC and can be pulled using the following command:
```bash
docker pull nvcr.io/nvidia/morpheus/morpheus-tritonserver-models:25.02
```

Those users who are working on training their own models have two options available:
1) Build the models container locally by running the following command from the root of the Morpheus repo:
```bash
./models/docker/build_container.sh
```

This option is good for users who have a model which has already been trained and is ready for deployment. For more information refer to the [README](./docker/README.md) in the `docker` directory.

2) Using the Triton Docker image directly, and mounting the `models` directory into the container. This option is good for users who are iterating on a single model and do not wish to build the entire container each time. The rest of this document covers using this option.

## Launching Triton

To launch Triton with one of the models in `triton-model-repo`, this entire repo must be volume mounted into the container. Once the entire repository is mounted, the Triton options: `--model-repository` and `--load-model` can be selectively used to choose which models to load. The following are several examples on launching Triton with different models and different setups:

### Load `sid-minibert-onnx` Model with Default Triton Image

```bash
docker run --rm --gpus=all -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $PWD:/models --name tritonserver nvcr.io/nvidia/tritonserver:22.08-py3 tritonserver --model-repository=/models/triton-model-repo --exit-on-error=false --model-control-mode=explicit --load-model sid-minibert-onnx
```

### Load `abp-nvsmi-xgb` Model with FIL Back-end Triton

```bash
docker run --rm --gpus=all -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $PWD:/models --name tritonserver triton_fil tritonserver --model-repository=/models/triton-model-repo --exit-on-error=false --model-control-mode=explicit --load-model abp-nvsmi-xgb
```

### Load `sid-minibert-trt` Model with Default Triton Image from Morpheus Repo

To load a TensorRT model, it first must be compiled with the `morpheus tools onnx-to-trt` utility. This utility requires additional packages to be installed. From the root of the Morpheus repo, install them with:
```bash
conda env update --solver=libmamba -n morpheus --file conda/environments/model-utils_cuda-125_arch-$(arch).yaml
```

Then build the TensorRT model with (refer `triton-model-repo/sid-minibert-trt/1/README.md` for more info):
```bash
cd models/triton-model-repo/sid-minibert-trt/1
morpheus --log_level=info tools onnx-to-trt --input_model ../../sid-minibert-onnx/1/model.onnx --output_model ./model.plan --batches 1 8 --batches 1 16 --batches 1 32 --seq_length 256 --max_workspace_size 16000
```

Then launch Triton:

```bash
docker run --rm --gpus=all -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $PWD/models:/models --name tritonserver nvcr.io/nvidia/tritonserver:22.08-py3 tritonserver --model-repository=/models/triton-model-repo --exit-on-error=false --model-control-mode=explicit --load-model sid-minibert-trt
```
