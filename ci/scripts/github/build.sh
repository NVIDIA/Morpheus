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

CMAKE_FLAGS="${CMAKE_BUILD_ALL_FEATURES}"
CMAKE_FLAGS="${CMAKE_FLAGS} -DMORPHEUS_PYTHON_BUILD_WHEEL=ON"
CMAKE_FLAGS="${CMAKE_FLAGS} -DMORPHEUS_PYTHON_BUILD_STUBS=OFF"
CMAKE_FLAGS="${CMAKE_FLAGS} -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON"

rapids-logger "Configuring cmake for Morpheus with ${CMAKE_FLAGS}"
cmake ${CMAKE_FLAGS} .

rapids-logger "Building Morpheus"
cmake --build ${BUILD_DIR} --parallel ${PARALLEL_LEVEL}

log_sccache_stats

rapids-logger "Archiving results"
tar cfj "${WORKSPACE_TMP}/wheel-${REAL_ARCH}.tar.bz" ${BUILD_DIR}/python/morpheus/dist ${BUILD_DIR}/python/morpheus_llm/dist ${BUILD_DIR}/python/morpheus_dfp/dist

MORPHEUS_LIBS=($(find ${MORPHEUS_ROOT}/${BUILD_DIR}/python/morpheus/morpheus/_lib -name "*.so" -exec realpath --relative-to ${MORPHEUS_ROOT} {} \;) \
                $(find ${MORPHEUS_ROOT}/${BUILD_DIR}/python/morpheus_llm/morpheus_llm/_lib -name "*.so" -exec realpath --relative-to ${MORPHEUS_ROOT} {} \;) \
                $(find ${MORPHEUS_ROOT}/examples -name "*.so" -exec realpath --relative-to ${MORPHEUS_ROOT} {} \;))
tar cfj "${WORKSPACE_TMP}/morhpeus_libs-${REAL_ARCH}.tar.bz" "${MORPHEUS_LIBS[@]}"

CPP_TESTS=($(find ${MORPHEUS_ROOT}/${BUILD_DIR}/python/morpheus/morpheus/_lib/tests -name "*.x" -exec realpath --relative-to ${MORPHEUS_ROOT} {} \;) \
            $(find ${MORPHEUS_ROOT}/${BUILD_DIR}/python/morpheus_llm/morpheus_llm/_lib/tests -name "*.x" -exec realpath --relative-to ${MORPHEUS_ROOT} {} \;))
tar cfj "${WORKSPACE_TMP}/cpp_tests-${REAL_ARCH}.tar.bz" "${CPP_TESTS[@]}"

rapids-logger "Pushing results to ${DISPLAY_ARTIFACT_URL}"
set_job_summary_preamble
upload_artifact "${WORKSPACE_TMP}/wheel-${REAL_ARCH}.tar.bz"
upload_artifact "${WORKSPACE_TMP}/morhpeus_libs-${REAL_ARCH}.tar.bz"
upload_artifact "${WORKSPACE_TMP}/cpp_tests-${REAL_ARCH}.tar.bz"

rapids-logger "Success"
exit 0
