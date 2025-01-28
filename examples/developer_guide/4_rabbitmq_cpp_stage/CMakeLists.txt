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

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

list(APPEND CMAKE_MESSAGE_CONTEXT "4_rabbitmq_cpp_stage")

# Set the cache to be the same to allow for CCache to be used effectively
set(MORPHEUS_CACHE_DIR "${CMAKE_SOURCE_DIR}/.cache" CACHE PATH "Directory to contain all CPM and CCache data")
mark_as_advanced(MORPHEUS_CACHE_DIR)

# Add the Conda environment to the prefix path and add the CMake files
list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(4_rabbitmq_cpp_stage
  VERSION 25.02.00
  LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the option prefix to match the outer project before including. Must be before find_package(morpheus)
set(OPTION_PREFIX "MORPHEUS")

# Set the policy to allow for CMP0144, avoids warning about MORPHEUS_ROOT being set
cmake_policy(SET CMP0144 NEW)

find_package(morpheus REQUIRED)

morpheus_utils_initialize_cpm(MORPHEUS_CACHE_DIR)

# Ensure CPM is initialized
rapids_cpm_init()

morpheus_utils_python_configure()

include(cmake/dependencies.cmake)

morpheus_utils_create_python_package(morpheus_rabbit
  PROJECT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

add_subdirectory(src/rabbitmq_cpp_stage/_lib)

# Complete the python package
if(MORPHEUS_PYTHON_INPLACE_BUILD)
  list(APPEND extra_args "IS_INPLACE")
endif()

if(TARGET morpheus-package-install)
  list(APPEND extra_args "PYTHON_DEPENDENCIES" "morpheus-package-install")
endif()

morpheus_utils_build_python_package(morpheus_rabbit ${extra_args})

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
