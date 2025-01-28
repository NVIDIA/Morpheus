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

list(APPEND CMAKE_MESSAGE_CONTEXT "dep")

morpheus_utils_initialize_cpm(MORPHEUS_CACHE_DIR)

# Show some setup variables (only prints if VERBOSE)
morpheus_utils_print_config()

# First, load the package_config functions
include(${CMAKE_CURRENT_LIST_DIR}/package_config/register_api.cmake)

# Load direct physical package dependencies first, so we fail early. Add all dependencies to our export set
rapids_find_package(Protobuf REQUIRED
  BUILD_EXPORT_SET ${PROJECT_NAME}-core-exports
  INSTALL_EXPORT_SET ${PROJECT_NAME}-core-exports
)

rapids_find_package(CUDAToolkit REQUIRED
  BUILD_EXPORT_SET ${PROJECT_NAME}-core-exports
  INSTALL_EXPORT_SET ${PROJECT_NAME}-core-exports
)

rapids_find_package(ZLIB
  BUILD_EXPORT_SET ${PROJECT_NAME}-core-exports
  INSTALL_EXPORT_SET ${PROJECT_NAME}-core-exports
)

if(MORPHEUS_BUILD_BENCHMARKS)
  # google benchmark
  # ================
  include(${rapids-cmake-dir}/cpm/gbench.cmake)
  rapids_cpm_gbench(
    BUILD_EXPORT_SET ${PROJECT_NAME}-core-exports
    INSTALL_EXPORT_SET ${PROJECT_NAME}-core-exports
  )
endif()

# gflags
# ======
rapids_find_package(gflags REQUIRED
  GLOBAL_TARGETS gflags
  BUILD_EXPORT_SET ${PROJECT_NAME}-exports
  INSTALL_EXPORT_SET ${PROJECT_NAME}-exports
)

# glog
# ====
morpheus_utils_configure_glog()

if(MORPHEUS_BUILD_TESTS)
  # google test
  # ===========
  include(${rapids-cmake-dir}/cpm/gtest.cmake)
  rapids_cpm_gtest(
    BUILD_EXPORT_SET ${PROJECT_NAME}-core-exports
    INSTALL_EXPORT_SET ${PROJECT_NAME}-core-exports
  )
endif()

# Include dependencies based on components being built
if(MORPHEUS_BUILD_MORPHEUS_CORE)
  include(dependencies_core)
endif()

if(MORPHEUS_BUILD_MORPHEUS_LLM)
  include(dependencies_llm)
endif()

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
