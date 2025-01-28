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

# It is assumed that this script is executed from the root of the repo directory by conda-build
# (https://conda-forge.org/docs/maintainer/knowledge_base.html#using-cmake)

# Need to ensure this value is set before checking it in the if block
MORPHEUS_SUPPORT_DOCA=${MORPHEUS_SUPPORT_DOCA:-OFF}

# This will store all of the cmake args. Make sure to prepend args to allow
# incoming values to overwrite them
CMAKE_ARGS=${CMAKE_ARGS:-""}

export CCACHE_BASEDIR=$(realpath ${SRC_DIR}/..)
export USE_SCCACHE=${USE_SCCACHE:-""}

# Check for some mrc environment variables. Append to front of args to allow users to overwrite them
if [[ -n "${MORPHEUS_CACHE_DIR}" ]]; then
   # Set the cache variable, then set the Staging prefix to allow for host searching
   CMAKE_ARGS="-DMORPHEUS_CACHE_DIR=${MORPHEUS_CACHE_DIR} ${CMAKE_ARGS}"

   # Double check that the cache dir has been created
   mkdir -p ${MORPHEUS_CACHE_DIR}
fi

if [[ ${MORPHEUS_SUPPORT_DOCA} == @(TRUE|ON) ]]; then
   CMAKE_ARGS="-DMORPHEUS_SUPPORT_DOCA=ON ${CMAKE_ARGS}"

   # Set the CMAKE_CUDA_ARCHITECTURES to just 80;86 since that is what DOCA supports for now
   CMAKE_CUDA_ARCHITECTURES="80;86"

   echo "MORPHEUS_SUPPORT_DOCA is ON. Setting CMAKE_CUDA_ARCHITECTURES to supported values: '${CMAKE_CUDA_ARCHITECTURES}'"
fi

# enable all functional blocks
CMAKE_ARGS="-DMORPHEUS_BUILD_MORPHEUS_CORE=ON ${CMAKE_ARGS}"
CMAKE_ARGS="-DMORPHEUS_BUILD_MORPHEUS_LLM=ON ${CMAKE_ARGS}"
CMAKE_ARGS="-DMORPHEUS_BUILD_MORPHEUS_DFP=ON ${CMAKE_ARGS}"

CMAKE_ARGS="-DCMAKE_MESSAGE_CONTEXT_SHOW=ON ${CMAKE_ARGS}"
CMAKE_ARGS="-DCMAKE_INSTALL_PREFIX=$PREFIX ${CMAKE_ARGS}"
CMAKE_ARGS="-DCMAKE_INSTALL_LIBDIR=lib ${CMAKE_ARGS}"
CMAKE_ARGS="-DBUILD_SHARED_LIBS=ON ${CMAKE_ARGS}"
CMAKE_ARGS="-DMORPHEUS_USE_CONDA=ON ${CMAKE_ARGS}"
CMAKE_ARGS="-DMORPHEUS_USE_CCACHE=ON ${CMAKE_ARGS}"
CMAKE_ARGS="-DMORPHEUS_PYTHON_BUILD_STUBS=${MORPHEUS_PYTHON_BUILD_STUBS=-"ON"} ${CMAKE_ARGS}"
CMAKE_ARGS="-DMORPHEUS_PYTHON_INPLACE_BUILD=ON ${CMAKE_ARGS}"
CMAKE_ARGS="-DMORPHEUS_PYTHON_BUILD_WHEEL=ON ${CMAKE_ARGS}"
CMAKE_ARGS="-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON ${CMAKE_ARGS}"
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES=-"RAPIDS"} ${CMAKE_ARGS}"
CMAKE_ARGS="-DPython_EXECUTABLE=${PYTHON} ${CMAKE_ARGS}"
CMAKE_ARGS="-DPYTHON_EXECUTABLE=${PYTHON} ${CMAKE_ARGS}" # for pybind11
CMAKE_ARGS="--log-level=VERBOSE ${CMAKE_ARGS}"

if [[ "${USE_SCCACHE}" == "1" ]]; then
   CMAKE_ARGS="-DCCACHE_PROGRAM_PATH=$(which sccache) ${CMAKE_ARGS}"
fi

echo "CC          : ${CC}"
echo "CXX         : ${CXX}"
echo "CUDAHOSTCXX : ${CUDAHOSTCXX}"
echo "CUDA        : ${CUDA}"
echo "CMAKE_ARGS  : ${CMAKE_ARGS}"

echo "========Begin Env========"
env
echo "========End Env========"

BUILD_DIR="build-conda"

# Check if the build directory already exists. And if so, delete the
# CMakeCache.txt and CMakeFiles to ensure a clean configuration
if [[ -d "./${BUILD_DIR}" ]]; then
   echo "Deleting old CMake files at ./${BUILD_DIR}"
   rm -rf "./${BUILD_DIR}/CMakeCache.txt"
   rm -rf "./${BUILD_DIR}/CMakeFiles"
fi

# Run configure
cmake -B ${BUILD_DIR} \
   ${CMAKE_ARGS} \
   --log-level=verbose \
   .

# Build the components
cmake --build ${BUILD_DIR} -j${PARALLEL_LEVEL:-$(nproc)} --target install

# Install just the python wheel components
${PYTHON} -m pip install -vv ${BUILD_DIR}/python/morpheus/dist/*.whl
${PYTHON} -m pip install -vv ${BUILD_DIR}/python/morpheus_llm/dist/*.whl
${PYTHON} -m pip install -vv ${BUILD_DIR}/python/morpheus_dfp/dist/*.whl
