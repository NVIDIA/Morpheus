#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

HOST_ARCH=$(dpkg --print-architecture)
DOCKER_TARGET_ARCH=${DOCKER_TARGET_ARCH:-${HOST_ARCH}}

if [ ${DOCKER_TARGET_ARCH} != ${HOST_ARCH} ]; then
    echo -n "Performing cross-build for ${DOCKER_TARGET_ARCH} on ${HOST_ARCH}, please ensure qemu is installed, "
    echo "details in ${MORPHEUS_ROOT}/external/utilities/ci/runner/README.md"
fi

# Determine the relative path from $PWD to $MORPHEUS_ROOT
MORPHEUS_ROOT_HOST=${MORPHEUS_ROOT_HOST:-"$(realpath --relative-to=${PWD} ${MORPHEUS_ROOT})"}

FULL_VERSION=$(git describe --tags --abbrev=0)
MAJOR_VERSION=$(echo ${FULL_VERSION} | awk '{split($0, a, "[v.]"); print a[2]}')
MINOR_VERSION=$(echo ${FULL_VERSION} | awk '{split($0, a, "."); print a[2]}')
SHORT_VERSION=${MAJOR_VERSION}.${MINOR_VERSION}

# Build args
FROM_IMAGE=${FROM_IMAGE:-"nvcr.io/nvidia/tritonserver"}
FROM_IMAGE_TAG=${FROM_IMAGE_TAG:-"24.09-py3"}

DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME:-"nvcr.io/nvidia/morpheus/morpheus-tritonserver-models"}
DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG:-"${SHORT_VERSION}"}

DOCKER_EXTRA_ARGS=${DOCKER_EXTRA_ARGS:-""}

# Ensure all models are fetched
"${MORPHEUS_ROOT}/scripts/fetch_data.py" fetch models

# Build the docker arguments
DOCKER_ARGS="-t ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg FROM_IMAGE=${FROM_IMAGE}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg FROM_IMAGE_TAG=${FROM_IMAGE_TAG}"
DOCKER_ARGS="${DOCKER_ARGS} --platform=linux/${DOCKER_TARGET_ARCH}"
DOCKER_ARGS="${DOCKER_ARGS} --network=host"

# Last add any extra args (duplicates override earlier ones)
DOCKER_ARGS="${DOCKER_ARGS} ${DOCKER_EXTRA_ARGS}"

# Export buildkit variable
export DOCKER_BUILDKIT=1

echo "Building morpheus:${DOCKER_TAG} with args..."
echo "   FROM_IMAGE           : ${FROM_IMAGE}"
echo "   FROM_IMAGE           : ${FROM_IMAGE_TAG}"

echo ""
echo "   COMMAND: docker build ${DOCKER_ARGS} -f ${SCRIPT_DIR}/Dockerfile ."
echo "   Note: add '--progress plain' to DOCKER_EXTRA_ARGS to show all container build output"

docker build ${DOCKER_ARGS} -f ${SCRIPT_DIR}/Dockerfile .
