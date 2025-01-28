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

export WORKSPACE_TMP="$(pwd)/ws_tmp"
mkdir -p ${WORKSPACE_TMP}

if [[ "${USE_HOST_GIT}" == "1" ]]; then
    cd Morpheus/
else
    git clone ${GIT_URL} Morpheus
    cd Morpheus/
    git checkout ${GIT_BRANCH}
    git pull
    git checkout ${GIT_COMMIT}
    git fetch --tags
    git submodule update --init --recursive
fi

export MORPHEUS_ROOT=$(pwd)
export WORKSPACE=${MORPHEUS_ROOT}
export LOCAL_CI=1
unset CMAKE_CUDA_COMPILER_LAUNCHER
unset CMAKE_CXX_COMPILER_LAUNCHER
unset CMAKE_C_COMPILER_LAUNCHER

if [[ "${STAGE}" != "bash" ]]; then
    ${MORPHEUS_ROOT}/ci/scripts/github/${STAGE}.sh
fi
