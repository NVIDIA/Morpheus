#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

# Overwrite HOME to WORKSPACE
export HOME="$WORKSPACE"

# Install gpuCI tools
curl -s https://raw.githubusercontent.com/rapidsai/gpuci-tools/main/install.sh | bash
source ~/.bashrc
cd ~

# Show env
gpuci_logger "Exposing current environment..."
env

# Select dockerfile based on matrix var
DOCKERFILE="docker/Dockerfile"
DOCKER_CONTEXT="."
gpuci_logger "Using Dockerfile: ${DOCKERFILE}"
gpuci_logger "Using Context: ${DOCKER_CONTEXT}"

# Debug output selected dockerfile
gpuci_logger ">>>> BEGIN Dockerfile <<<<"
cat ${DOCKERFILE}
gpuci_logger ">>>> END Dockerfile <<<<"

# Get build info ready
gpuci_logger "Preparing build config..."
BUILD_TAG="cuda${CUDA}-${LINUX_VER}-py${PYTHON_VER}"

# Setup initial BUILD_ARGS
BUILD_ARGS="--no-cache \
  --build-arg CUDA_VER=${CUDA} \
  --build-arg LINUX_VER=${LINUX_VER} \
  --build-arg PYTHON_VER=${PYTHON_VER}"

# Ouput build config
gpuci_logger "Build config info..."
echo "Build args: ${BUILD_ARGS}"
gpuci_logger "Docker build command..."
echo "docker build --pull -t morpheus:${BUILD_TAG} ${BUILD_ARGS} -f ${DOCKERFILE} ${DOCKER_CONTEXT}"

# Build image
gpuci_logger "Starting build..."
docker build --pull -t morpheus:${BUILD_TAG} ${BUILD_ARGS} -f ${DOCKERFILE} ${DOCKER_CONTEXT}

# List image info
gpuci_logger "Displaying image info..."
docker images morpheus:${BUILD_TAG}
