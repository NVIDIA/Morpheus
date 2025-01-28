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

_FLAGS=()
_FLAGS+=("-B" "${BUILD_DIR}")
_FLAGS+=("-G" "Ninja")
_FLAGS+=("-DCMAKE_MESSAGE_CONTEXT_SHOW=ON")
_FLAGS+=("-DMORPHEUS_CUDA_ARCHITECTURES=RAPIDS")
_FLAGS+=("-DMORPHEUS_USE_CONDA=ON")
_FLAGS+=("-DMORPHEUS_USE_CCACHE=ON")
_FLAGS+=("-DMORPHEUS_PYTHON_INPLACE_BUILD=OFF")
_FLAGS+=("-DMORPHEUS_PYTHON_BUILD_STUBS=ON")
_FLAGS+=("-DMORPHEUS_BUILD_BENCHMARKS=ON")
_FLAGS+=("-DMORPHEUS_BUILD_EXAMPLES=ON")
_FLAGS+=("-DMORPHEUS_BUILD_TESTS=ON")
_FLAGS+=("-DMORPHEUS_BUILD_MORPHEUS_LLM=ON")
_FLAGS+=("-DMORPHEUS_BUILD_MORPHEUS_DFP=ON")
if [[ "${LOCAL_CI}" == "" ]]; then
    _FLAGS+=("-DCCACHE_PROGRAM_PATH=$(which sccache)")
fi
export CMAKE_BUILD_ALL_FEATURES="${_FLAGS[@]}"
unset _FLAGS

if [[ ${MORPHEUS_SUPPORT_DOCA} == @(TRUE|ON) ]]; then
    export CMAKE_BUILD_ALL_FEATURES="${CMAKE_BUILD_ALL_FEATURES} -DMORPHEUS_SUPPORT_DOCA=ON"
fi
