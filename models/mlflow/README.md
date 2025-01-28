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

# MLflow Triton

MLflow plugin for deploying your models from MLflow to Triton Inference Server. Scripts
are included for publishing TensorRT, ONNX and FIL models to your MLflow Model Registry.

## Requirements

* MLflow (tested on 2.11.3)
* Python (tested on 3.11)

## Install Triton Docker Image

Before you can use the Triton Docker image you must install
[Docker](https://docs.docker.com/engine/install). If you plan on using
a GPU for inference you must also install the [NVIDIA Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). DGX users should
follow [Preparing to use NVIDIA
Containers](http://docs.nvidia.com/deeplearning/dgx/preparing-containers/index.html).

Pull the image using the following command.

```
docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3
```

Where \<xx.yy\> is the version of Triton that you want to pull.

## Set up your Triton Model Repository
Create a directory on your host machine that will serve as your Triton model repository. This directory will contain the models to be used by Morpheus and will be volume mounted to your Triton Inference Server container.

Example:

```
mkdir -p /opt/triton_models
```

## Download Morpheus reference models

The Morpheus reference models can be found in the [Morpheus](https://github.com/nv-morpheus/Morpheus) repo. A script is provided to fetch the models using git-lfs due to size.
Before running the MLflow plugin container, you can fetch the models and mount them to the local path on your host (for example, `/opt/triton_models`).

```bash
git clone https://github.com/nv-morpheus/Morpheus.git morpheus
cd morpheus
scripts/fetch_data.py fetch models
cp -RL models /opt/triton_models
```

## Start Triton Inference Server in EXPLICIT mode

Use the following command to run Triton with our model repository you just created. The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) must be installed for Docker to recognize the GPUs. The `--gpus=1` flag indicates that the GPU with ID `1` should be made available to Triton for inferencing.

```bash
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /opt/triton_models:/models nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/models --model-control-mode=explicit
```

## MLflow container

Build MLflow image, from the root of the Morpheus repo:

```bash
cd models/mlflow
docker build -t mlflow-triton-plugin:latest -f docker/Dockerfile .
```

Create an MLflow container with a volume mounting the Triton model repository:

```bash
docker run -it -v /opt/triton_models:/triton_models \
--env TRITON_MODEL_REPO=/triton_models \
--env MLFLOW_TRACKING_URI="http://localhost:5000" \
--gpus '"device=0"' \
--net=host \
--rm \
-d mlflow-triton-plugin:latest
```

Open Bash shell in container:

```bash
docker exec -it <container_name> bash
```

## Start MLflow server

```
nohup mlflow server --backend-store-uri sqlite:////tmp/mlflow-db.sqlite --default-artifact-root /mlflow/artifacts --host 0.0.0.0 &
```

## Publish reference models to MLflow

The `publish_model_to_mlflow` script is used to publish `triton` flavor models to MLflow. A `triton` flavor model is a directory containing the model files following the [model layout](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md#repository-layout). Below is an example usage:

```
python publish_model_to_mlflow.py \
	--model_name sid-minibert-onnx \
	--model_directory /triton_models/triton-model-repo/sid-minibert-onnx \
    --flavor triton
```

## Deployments

The Triton `mlflow-triton-plugin` is installed on this container and can be used to deploy your models from MLflow to Triton Inference Server. The following are examples of how the plugin is used with the `sid-minibert-onnx` model that we published to MLflow above. For more information about the
`mlflow-triton-plugin`, refer to Triton's [documentation](https://github.com/triton-inference-server/server/tree/r24.03/deploy/mlflow-triton-plugin)

### Create Deployment

To create a deployment use the following command

##### CLI
```
mlflow deployments create -t triton --flavor triton --name sid-minibert-onnx -m "models:/sid-minibert-onnx/1"
```

##### Python API
```
from mlflow.deployments import get_deploy_client
client = get_deploy_client('triton')
client.create_deployment("sid-minibert-onnx", "models:/sid-minibert-onnx/1", flavor="triton")
```

### Delete Deployment

##### CLI
```
mlflow deployments delete -t triton --name sid-minibert-onnx
```

##### Python API
```
from mlflow.deployments import get_deploy_client
client = get_deploy_client('triton')
client.delete_deployment("sid-minibert-onnx")
```

### Update Deployment

##### CLI
```
mlflow deployments update -t triton --flavor triton --name sid-minibert-onnx -m "models:/sid-minibert-onnx/1"
```

##### Python API
```
from mlflow.deployments import get_deploy_client
client = get_deploy_client('triton')
client.update_deployment("sid-minibert-onnx", "models:/sid-minibert-onnx/1", flavor="triton")
```

### List Deployments

##### CLI
```
mlflow deployments list -t triton
```

##### Python API
```
from mlflow.deployments import get_deploy_client
client = get_deploy_client('triton')
client.list_deployments()
```

### Get Deployment

##### CLI
```
mlflow deployments get -t triton --name sid-minibert-onnx
```

##### Python API
```
from mlflow.deployments import get_deploy_client
client = get_deploy_client('triton')
client.get_deployment("sid-minibert-onnx")
```
