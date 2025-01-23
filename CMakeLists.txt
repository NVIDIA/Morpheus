# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

list(APPEND CMAKE_MESSAGE_CONTEXT "morpheus")

# Global options (Keep sorted!)
option(BUILD_SHARED_LIBS "Default value for whether or not to build shared or static libraries" ON)
option(MORPHEUS_BUILD_BENCHMARKS "Whether or not to build benchmarks" OFF)
option(MORPHEUS_BUILD_DOCS "Enable building of API documentation" OFF)
option(MORPHEUS_BUILD_EXAMPLES "Whether or not to build examples" OFF)
option(MORPHEUS_BUILD_MORPHEUS_CORE "Whether or not to build morpheus_core" ON)
option(MORPHEUS_BUILD_MORPHEUS_DFP "Whether or not to build morpheus_dfp" ON)
option(MORPHEUS_BUILD_MORPHEUS_LLM "Whether or not to build morpheus_llm" ON)
option(MORPHEUS_BUILD_TESTS "Whether or not to build tests" OFF)
option(MORPHEUS_ENABLE_DEBUG_INFO "Enable printing debug information" OFF)
option(MORPHEUS_PYTHON_BUILD_STUBS "Whether or not to generated .pyi stub files for C++ Python modules. Disable to avoid requiring loading the NVIDIA GPU Driver during build" ON)
option(MORPHEUS_PYTHON_BUILD_WHEEL "Whether or not to build the morpheus .whl file" OFF)
option(MORPHEUS_PYTHON_INPLACE_BUILD "Whether or not to copy built python modules back to the source tree for debug purposes." OFF)
option(MORPHEUS_PYTHON_PERFORM_INSTALL "Whether or not to automatically `pip install` any built python library. WARNING: This may overwrite any existing installation of the same name." OFF)
option(MORPHEUS_SUPPORT_DOCA "Whether or not to build doca-related elements of morpheus" OFF)
option(MORPHEUS_USE_CCACHE "Enable caching compilation results with ccache" OFF)
option(MORPHEUS_USE_CLANG_TIDY "Enable running clang-tidy as part of the build process" OFF)
option(MORPHEUS_USE_CONDA "Enables finding dependencies via conda instead of vcpkg. Note: This will disable vcpkg. All dependencies must be installed first in the conda environment" OFF)
option(MORPHEUS_USE_IWYU "Enable running include-what-you-use as part of the build process" OFF)

set(MORPHEUS_PY_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/wheel" CACHE STRING "Location to install the python directory")

set(MORPHEUS_RAPIDS_VERSION "25.02" CACHE STRING "Sets default versions for RAPIDS libraries.")
set(MORPHEUS_CACHE_DIR "${CMAKE_SOURCE_DIR}/.cache" CACHE PATH "Directory to contain all CPM and CCache data")
mark_as_advanced(MORPHEUS_CACHE_DIR)

enable_testing()

if(MORPHEUS_USE_IWYU AND MORPHEUS_USE_CCACHE)
  message(FATAL_ERROR "MORPHEUS_USE_IWYU and MORPHEUS_USE_CCACHE cannot be set simultaneously")
endif()

if (MORPHEUS_BUILD_DOCS AND NOT MORPHEUS_BUILD_MORPHEUS_LLM)
  message(FATAL_ERROR "MORPHEUS_BUILD_MORPHEUS_LLM must be ON if MORPHEUS_BUILD_DOCS is ON")
endif()

# MRC CMake path and module extensions
set(MORPHEUS_CMAKE_MODULE_PATH_EXTENSIONS
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/package_search"
  "${CMAKE_CURRENT_SOURCE_DIR}/external/utilities/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/external/utilities/cmake/morpheus_utils/package_search"
)

set(MORPHEUS_CMAKE_PREFIX_PATH_EXTENSIONS
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
)

# Prepend path and prefix updates so they take priority in this scope.
list(PREPEND CMAKE_MODULE_PATH "${MORPHEUS_CMAKE_MODULE_PATH_EXTENSIONS}")
list(PREPEND CMAKE_PREFIX_PATH "${MORPHEUS_CMAKE_PREFIX_PATH_EXTENSIONS}")

# Force the MORPHEUS_UTILS_RAPIDS_VERSION to match our value
set(MORPHEUS_UTILS_RAPIDS_VERSION ${MORPHEUS_RAPIDS_VERSION} CACHE STRING "" FORCE)

# Load morpheus utils and update CMake paths
include(morpheus_utils/load)

morpheus_utils_initialize_package_manager(
  MORPHEUS_USE_CONDA
  BUILD_SHARED_LIBS
)

# Initialize CUDA
# This is a two-step process. We need to call morpheus_utils_initialize_cuda_arch which in turn calls
# rapids_cuda_init_architectures prior to calling project(). This is because rapids_cuda_init_architectures defines a
# `CMAKE_PROJECT_<PROJECT-NAME>_INCLUDE` hook which is invoked by the project() call. This hook is what allows us to
# set `CMAKE_CUDA_ARCHITECTURES=rapids` when performing a release build which will be expanded to the current list of
# supported architectures by our version of rapids.
#
# After the call to project() we can then call morpheus_utils_enable_cuda() which will set some CUDA+clang settings
# which can only be performed after calling project(), but which must be set prior to calling enable_language(CUDA)
if(DEFINED MORPHEUS_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "${MORPHEUS_CUDA_ARCHITECTURES}")
endif()

morpheus_utils_initialize_cuda_arch(morpheus)

# Project definition
# Note intentionally excluding CUDA from the LANGUAGES list allowing us to set some clang specific settings later when
# we call morpheus_utils_enable_cuda()
project(morpheus
  VERSION 25.02.00
  LANGUAGES C CXX
)

# This sets some clang specific settings for CUDA prior to calling enable_language(CUDA)
morpheus_utils_enable_cuda()

morpheus_utils_initialize_install_prefix(MORPHEUS_USE_CONDA)

rapids_cmake_write_version_file(${CMAKE_BINARY_DIR}/autogenerated/include/morpheus/version.hpp)

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN")

# Setup cache before dependencies
# Configure CCache if requested
include(environment/init_ccache)

# Disable exporting compile commands for dependencies
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

# Create a custom target to allow preparing for style checks
add_custom_target(${PROJECT_NAME}_style_checks
  COMMENT "Building dependencies for style checks"
)

# CMAKE_LIBRARY_ARCHITECTURE is needed to find DOCA dependencies
if(NOT DEFINED CMAKE_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_SYSTEM_NAME}-${CMAKE_CXX_COMPILER_ID}")
  string(TOLOWER "${CMAKE_LIBRARY_ARCHITECTURE}" CMAKE_LIBRARY_ARCHITECTURE)
  message(STATUS "Setting CMAKE_LIBRARY_ARCHITECTURE to ${CMAKE_LIBRARY_ARCHITECTURE}")
endif()

# Configure all dependencies
include(dependencies)

# Enable for all first party code
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# To make it easier for CI to find output files, set the default executable suffix to .x if not set
if("${CMAKE_EXECUTABLE_SUFFIX}" STREQUAL "")
  set(CMAKE_EXECUTABLE_SUFFIX ".x")
endif()

# ###################################
# - Post dependencies setup --------
morpheus_utils_compiler_set_defaults(MORPHEUS_USE_CLANG_TIDY)

# Setup IWYU if enabled
include(environment/init_iwyu)

add_subdirectory(python)

if(MORPHEUS_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(MORPHEUS_BUILD_DOCS)
  add_subdirectory(docs)
endif()

if(MORPHEUS_ENABLE_DEBUG_INFO)
  morpheus_utils_print_all_targets()

  morpheus_utils_print_target_properties(
    TARGETS
      morpheus
    WRITE_TO_FILE
  )

  morpheus_utils_print_global_properties(
    WRITE_TO_FILE
  )
endif()

# Cleanup the environment after we exit this scope
list(REMOVE_ITEM CMAKE_PREFIX_PATH "${MORPHEUS_CMAKE_PREFIX_PATH_EXTENSIONS}")
list(REMOVE_ITEM CMAKE_MODULE_PATH "${MORPHEUS_CMAKE_MODULE_PATH_EXTENSIONS}")

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
