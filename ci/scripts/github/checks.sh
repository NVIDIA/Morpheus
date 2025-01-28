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

set -e

source ${WORKSPACE}/ci/scripts/github/common.sh
source ${WORKSPACE}/ci/scripts/github/morpheus_env.sh
source ${WORKSPACE}/ci/scripts/github/cmake_all.sh

rapids-dependency-file-generator \
  --output conda \
  --file-key build \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=${REAL_ARCH};py=${RAPIDS_PY_VERSION}" | tee "${WORKSPACE_TMP}/env.yaml"

update_conda_env "${WORKSPACE_TMP}/env.yaml"

log_toolchain

cd ${MORPHEUS_ROOT}

# Fetching the base branch will try methods that might fail, then fallback to one that does, set +e for this section
set +e
fetch_base_branch
set -e

rapids-logger "Configuring cmake for Morpheus"
CMAKE_FLAGS="${CMAKE_BUILD_ALL_FEATURES}"
CMAKE_FLAGS="${CMAKE_FLAGS} -DMORPHEUS_PYTHON_BUILD_STUBS=OFF"
export CMAKE_FLAGS="${CMAKE_FLAGS} -DMORPHEUS_PYTHON_INPLACE_BUILD=ON"

cmake ${CMAKE_FLAGS} .

rapids-logger "Building Morpheus"
cmake --build ${BUILD_DIR} --parallel ${PARALLEL_LEVEL}

log_sccache_stats

rapids-logger "Installing Morpheus"
pip install ./python/morpheus
pip install ./python/morpheus_llm
pip install ./python/morpheus_dfp

rapids-logger "Checking copyright headers"
python ${MORPHEUS_ROOT}/ci/scripts/copyright.py --verify-apache-v2 --git-diff-commits ${CHANGE_TARGET} ${GIT_COMMIT}

rapids-logger "Running Python style checks"
${MORPHEUS_ROOT}/ci/scripts/python_checks.sh

rapids-logger "Checking versions"
${MORPHEUS_ROOT}/ci/scripts/version_checks.sh

rapids-logger "Runing C++ style checks"
${MORPHEUS_ROOT}/ci/scripts/cpp_checks.sh

rapids-logger "Runing Documentation checks"
${MORPHEUS_ROOT}/ci/scripts/documentation_checks.sh
