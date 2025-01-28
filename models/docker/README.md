<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Morpheus Triton Server Models Container

The Morpheus Triton Server Models Container builds upon the [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server) container by adding the Morpheus pre-trained models.

## Building the Container
To build the container with the default arguments, run the following command from the root of the Morpheus repository:
```bash
./models/docker/build_container.sh
```

This will build a container tagged as `nvcr.io/nvidia/morpheus/morpheus-tritonserver-models:<Morpheus Version>`.

### Environment Variable Arguments
The `build_container.sh` script accepts the following environment variables to customize the build:
- `MORPHEUS_ROOT`: File path to the root of the Morpheus repository, if undefined the script will infer the value based on the script's location.
- `MORPHEUS_ROOT_HOST`: Relative path from the script working directory to the root of the Morpheus repository on the host. This should not need to be set so long as `MORPHEUS_ROOT` is set correctly.
- `FROM_IMAGE`: The base Triton Inference Server container image to use, defaults to `nvcr.io/nvidia/tritonserver`.
- `FROM_IMAGE_TAG`: The tag of the base Triton Inference Server container image to use.
- `DOCKER_IMAGE_NAME`: The name of the resulting container image, defaults to `nvcr.io/nvidia/morpheus/morpheus-tritonserver-models`.
- `DOCKER_IMAGE_TAG`: The tag of the resulting container image, defaults to the current Morpheus version.
- `DOCKER_EXTRA_ARGS`: Additional arguments to pass to the `docker build` command.
