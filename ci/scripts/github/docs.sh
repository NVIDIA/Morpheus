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
  --file-key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=${REAL_ARCH};py=${RAPIDS_PY_VERSION}" | tee "${WORKSPACE_TMP}/env.yaml"

update_conda_env "${WORKSPACE_TMP}/env.yaml"

download_artifact "wheel-${REAL_ARCH}.tar.bz"

tar xf "${WORKSPACE_TMP}/wheel-${REAL_ARCH}.tar.bz"

pip install ${MORPHEUS_ROOT}/${BUILD_DIR}/python/morpheus/dist/*.whl
pip install ${MORPHEUS_ROOT}/${BUILD_DIR}/python/morpheus_llm/dist/*.whl
pip install ${MORPHEUS_ROOT}/${BUILD_DIR}/python/morpheus_dfp/dist/*.whl

rapids-logger "Pulling LFS assets"
cd ${MORPHEUS_ROOT}

git lfs install
${MORPHEUS_ROOT}/scripts/fetch_data.py fetch docs examples

rapids-logger "Configuring for docs"
cmake ${CMAKE_BUILD_ALL_FEATURES} -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} -DMORPHEUS_PYTHON_BUILD_STUBS=OFF -DMORPHEUS_BUILD_DOCS=ON .

rapids-logger "Building docs"
cmake --build ${BUILD_DIR} --parallel ${PARALLEL_LEVEL} --target install
cmake --build ${BUILD_DIR} --parallel ${PARALLEL_LEVEL} --target morpheus_docs

rapids-logger "Checking documentation links"
cmake --build ${BUILD_DIR} --parallel ${PARALLEL_LEVEL} --target morpheus_docs_linkcheck

rapids-logger "Archiving the docs"
tar cfj "${WORKSPACE_TMP}/docs.tar.bz" ${BUILD_DIR}/docs/html

rapids-logger "Pushing results to ${DISPLAY_ARTIFACT_URL}"
set_job_summary_preamble
upload_artifact "${WORKSPACE_TMP}/docs.tar.bz"

rapids-logger "Success"
exit 0
