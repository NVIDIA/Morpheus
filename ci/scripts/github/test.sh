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
/usr/bin/nvidia-smi

update_conda_env "${WORKSPACE}/conda/environments/all_cuda-125_arch-${REAL_ARCH}.yaml"

log_toolchain

CMAKE_FLAGS="${CMAKE_BUILD_ALL_FEATURES}"
CMAKE_FLAGS="${CMAKE_FLAGS} -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON"
CMAKE_FLAGS="${CMAKE_FLAGS} -DMORPHEUS_PYTHON_BUILD_STUBS=ON"
CMAKE_FLAGS="${CMAKE_FLAGS} -DMORPHEUS_PYTHON_BUILD_WHEEL=OFF"
CMAKE_FLAGS="${CMAKE_FLAGS} -DMORPHEUS_PYTHON_PERFORM_INSTALL=ON"
CMAKE_FLAGS="${CMAKE_FLAGS} -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}"

rapids-logger "Configuring cmake for Morpheus with ${CMAKE_FLAGS}"
cmake ${CMAKE_FLAGS} .

rapids-logger "Building Morpheus"
cmake --build ${BUILD_DIR} --parallel ${PARALLEL_LEVEL} --target install

log_sccache_stats

rapids-logger "Checking Python stub files"

# Check for git diffs which would mean the build is out of sync with the repo
if [[ $(git status --short --untracked | grep .pyi) != "" ]]; then

    echo "The Python stubs (*.pyi) are out of sync with repo. Please rerun the build locally with "
    echo "'-DMORPHEUS_PYTHON_BUILD_STUBS=ON' and commit the stub files into the repo"
    git status
    exit 1
fi

CPP_TESTS=($(find ${MORPHEUS_ROOT}/${BUILD_DIR} -name "*.x"))

rapids-logger "Pulling LFS assets"

git lfs install
${MORPHEUS_ROOT}/scripts/fetch_data.py fetch tests validation

# Listing LFS-known files
rapids-logger "Listing LFS-known files"
git lfs ls-files

REPORTS_DIR="${WORKSPACE_TMP}/reports"
mkdir -p ${WORKSPACE_TMP}/reports

rapids-logger "Running C++ tests"
# Running the tests from the tests dir. Normally this isn't nescesary, however since
# we are testing the installed version of morpheus in site-packages and not the one
# in the repo dir, the pytest coverage module reports incorrect coverage stats.
pushd ${MORPHEUS_ROOT}/tests

TEST_RESULTS=0
for cpp_test in "${CPP_TESTS[@]}"; do
       test_name=$(basename ${cpp_test})
       rapids-logger "Running ${test_name}"
       set +e

       ${cpp_test} --gtest_output="xml:${REPORTS_DIR}/report_${test_name}.xml"
       TEST_RESULT=$?
       TEST_RESULTS=$(($TEST_RESULTS+$TEST_RESULT))

       set -e
done

rapids-logger "Running Python tests"
set +e

PYTEST_FLAGS=()
PYTEST_FLAGS+=("--run_slow")
PYTEST_FLAGS+=("--run_kafka")
if [[ "${REAL_ARCH}" == "x86_64" ]]; then
    PYTEST_FLAGS+=("--run_milvus")
    PYTEST_FLAGS+=("--fail_missing")
fi
PYTEST_FLAGS+=("--junit-xml=${REPORTS_DIR}/report_pytest.xml")
PYTEST_FLAGS+=("--cov=morpheus")
PYTEST_FLAGS+=("--cov-report term-missing")
PYTEST_FLAGS+=("--cov-report=xml:${REPORTS_DIR}/report_pytest_coverage.xml")

python -I -m pytest ${PYTEST_FLAGS[@]}

PYTEST_RESULTS=$?
TEST_RESULTS=$(($TEST_RESULTS+$PYTEST_RESULTS))

set -e
popd

rapids-logger "Archiving test reports"
cd $(dirname ${REPORTS_DIR})
tar cfj ${WORKSPACE_TMP}/test_reports-${REAL_ARCH}.tar.bz $(basename ${REPORTS_DIR})

rapids-logger "Pushing results to ${DISPLAY_ARTIFACT_URL}"
set_job_summary_preamble
upload_artifact ${WORKSPACE_TMP}/test_reports-${REAL_ARCH}.tar.bz

exit ${TEST_RESULTS}
