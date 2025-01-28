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

# Color variables
b="\033[0;36m"
g="\033[0;32m"
r="\033[0;31m"
e="\033[0;90m"
y="\033[0;33m"
x="\033[0m"

_UNDEF_VAR_ERROR_MSG="Use the dev/release scripts to set these automatically"

DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME:?"Must set \$DOCKER_IMAGE_NAME. ${_UNDEF_VAR_ERROR_MSG}"}
DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG:?"Must set \$DOCKER_IMAGE_TAG. ${_UNDEF_VAR_ERROR_MSG}"}

# DOCKER_ARGS are set by the dev/release scripts
# DOCKER_EXTRA_ARGS are optionally set by the user
DOCKER_ARGS=${DOCKER_ARGS:-""}
DOCKER_ARGS="${DOCKER_ARGS} --net=host --cap-add=sys_nice --platform=linux/${DOCKER_TARGET_ARCH} ${DOCKER_EXTRA_ARGS}"
DOCKER_EXTRA_ARGS=${DOCKER_EXTRA_ARGS:-""}

if [[ -n "${CPU_ONLY}" ]]; then
   echo -e "${b}Executing in CPU only mode${x}"
   DOCKER_ARGS="${DOCKER_ARGS} --runtime=runc"
else
    echo -e "${b}Executing in GPU mode${x}"
    DOCKER_ARGS="${DOCKER_ARGS} --runtime=nvidia --gpus=all"
fi

if [[ -n "${SSH_AUTH_SOCK}" ]]; then
   echo -e "${b}Setting up ssh-agent auth socket${x}"
   DOCKER_ARGS="${DOCKER_ARGS} -v $(readlink -f $SSH_AUTH_SOCK):/ssh-agent:ro -e SSH_AUTH_SOCK=/ssh-agent"
fi

echo -e "${g}Launching ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}...${x}"

# Enable command logging to show what is being executed
set -x
docker run ${DOCA_EXTRA_ARGS} --rm -ti ${DOCKER_ARGS} ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG} "${@:-bash}"

{ EXIT_CODE=$?; set +x; } 2>/dev/null

exit $EXIT_CODE
