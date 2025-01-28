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

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get the path to MORPHEUS_ROOT without altering the docker context (in case we are in a submodule)
pushd ${SCRIPT_DIR} &> /dev/null
export MORPHEUS_ROOT=${MORPHEUS_ROOT:-"$(git rev-parse --show-toplevel)"}
popd &> /dev/null

DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME:?"Must set \$DOCKER_IMAGE_NAME to build. Use the dev/release scripts to set these automatically"}
DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG:?"Must set \DOCKER_IMAGE_TAG to build. Use the dev/release scripts to set these automatically"}
DOCKER_TARGET=${DOCKER_TARGET:-"runtime"}

if [ "${DOCKER_TARGET_ARCH}" == "amd64" ]; then
    REAL_ARCH="x86_64"
elif [ "${DOCKER_TARGET_ARCH}" == "arm64" ]; then
    REAL_ARCH="aarch64"
else
    echo "Invalid DOCKER_TARGET_ARCH: ${DOCKER_TARGET_ARCH}"
    exit 1
fi

if [ ${REAL_ARCH} != $(arch) ]; then
    echo -n "Performing cross-build for ${REAL_ARCH} on $(arch), please ensure qemu is installed, "
    echo "details in ${MORPHEUS_ROOT}/external/utilities/ci/runner/README.md"
fi

DOCKER_EXTRA_ARGS=${DOCKER_EXTRA_ARGS:-""}

# Build args
CUDA_MAJOR_VER=${CUDA_MAJOR_VER:-12}
CUDA_MINOR_VER=${CUDA_MINOR_VER:-5}
CUDA_REV_VER=${CUDA_REV_VER:-1}
FROM_IMAGE=${FROM_IMAGE:-"nvidia/cuda"}
LINUX_DISTRO=${LINUX_DISTRO:-ubuntu}
LINUX_VER=${LINUX_VER:-22.04}
MORPHEUS_SUPPORT_DOCA=${MORPHEUS_SUPPORT_DOCA:-"OFF"}
MORPHEUS_BUILD_MORPHEUS_LLM=${MORPHEUS_BUILD_MORPHEUS_LLM:-"ON"}
MORPHEUS_BUILD_MORPHEUS_DFP=${MORPHEUS_BUILD_MORPHEUS_DFP:-"ON"}
PYTHON_VER=${PYTHON_VER:-3.10}

# Determine the relative path from $PWD to $MORPHEUS_ROOT
MORPHEUS_ROOT_HOST=${MORPHEUS_ROOT_HOST:-"$(realpath --relative-to=${PWD} ${MORPHEUS_ROOT})"}

# Build the docker arguments
DOCKER_ARGS="-t ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"
DOCKER_ARGS="${DOCKER_ARGS} --target ${DOCKER_TARGET}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg CUDA_MAJOR_VER=${CUDA_MAJOR_VER}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg CUDA_MINOR_VER=${CUDA_MINOR_VER}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg CUDA_REV_VER=${CUDA_REV_VER}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg FROM_IMAGE=${FROM_IMAGE}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg LINUX_DISTRO=${LINUX_DISTRO}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg LINUX_VER=${LINUX_VER}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg MORPHEUS_ROOT_HOST=${MORPHEUS_ROOT_HOST}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg MORPHEUS_SUPPORT_DOCA=${MORPHEUS_SUPPORT_DOCA}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg MORPHEUS_BUILD_MORPHEUS_LLM=${MORPHEUS_BUILD_MORPHEUS_LLM}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg MORPHEUS_BUILD_MORPHEUS_DFP=${MORPHEUS_BUILD_MORPHEUS_DFP}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg PYTHON_VER=${PYTHON_VER}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg REAL_ARCH=${REAL_ARCH}"
DOCKER_ARGS="${DOCKER_ARGS} --platform=linux/${DOCKER_TARGET_ARCH}"
DOCKER_ARGS="${DOCKER_ARGS} --network=host"

# Last add any extra args (duplicates override earlier ones)
DOCKER_ARGS="${DOCKER_ARGS} ${DOCKER_EXTRA_ARGS}"

# Export buildkit variable
export DOCKER_BUILDKIT=1

echo "Building morpheus:${DOCKER_TAG} with args..."
echo "   CUDA_MAJOR_VER             : ${CUDA_MAJOR_VER}"
echo "   CUDA_MINOR_VER             : ${CUDA_MINOR_VER}"
echo "   CUDA_REV_VER               : ${CUDA_REV_VER}"
echo "   FROM_IMAGE                 : ${FROM_IMAGE}"
echo "   LINUX_DISTRO               : ${LINUX_DISTRO}"
echo "   LINUX_VER                  : ${LINUX_VER}"
echo "   MORPHEUS_ROOT_HOST         : ${MORPHEUS_ROOT_HOST}"
echo "   MORPHEUS_SUPPORT_DOCA      : ${MORPHEUS_SUPPORT_DOCA}"
echo "   MORPHEUS_BUILD_MORPHEUS_LLM: ${MORPHEUS_BUILD_MORPHEUS_LLM}"
echo "   PYTHON_VER                 : ${PYTHON_VER}"
echo "   TARGET_ARCH                : ${DOCKER_TARGET_ARCH}"
echo "   REAL_ARCH                  : ${REAL_ARCH}"

echo ""
echo "   COMMAND: docker build ${DOCKER_ARGS} -f ${SCRIPT_DIR}/Dockerfile ."
echo "   Note: add '--progress plain' to DOCKER_EXTRA_ARGS to show all container build output"

docker build ${DOCKER_ARGS} -f ${SCRIPT_DIR}/Dockerfile .
