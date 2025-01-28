# =============================================================================
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# Need to load this first before calling find_package(doca REQUIRED)
morpheus_configure_libmd()
morpheus_configure_libbsd()

set(doca_ROOT "/opt/mellanox/doca")
find_package(doca REQUIRED)

add_library(morpheus_doca

  # Keep these sorted!
  src/doca_context.cpp
  src/doca_convert_kernel.cu
  src/doca_convert_stage.cpp
  src/doca_rx_pipe.cpp
  src/doca_rx_queue.cpp
  src/doca_semaphore.cpp
  src/doca_source_kernel.cu
  src/doca_source_stage.cpp
  src/packet_data_buffer.cpp
  src/rte_context.cpp
)

add_library(${PROJECT_NAME}::morpheus_doca ALIAS morpheus_doca)

target_include_directories(morpheus_doca
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${doca_ROOT}/include>
)

target_link_libraries(morpheus_doca
  PRIVATE
    doca::doca
    matx::matx
  PUBLIC
    ${PROJECT_NAME}::morpheus
)

# Ideally, we dont use glob here. But there is no good way to guarantee you dont miss anything like *.cpp
file(GLOB_RECURSE morpheus_doca_public_headers
  LIST_DIRECTORIES FALSE
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/include/morpheus/doca/*"
)

# Add headers to target sources file_set so they can be installed
# https://discourse.cmake.org/t/installing-headers-the-modern-way-regurgitated-and-revisited/3238/3
target_sources(morpheus_doca
  PUBLIC
    FILE_SET public_headers
    TYPE HEADERS
    BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
    FILES
      ${morpheus_doca_public_headers}
)

target_compile_definitions(morpheus_doca
  PUBLIC
    DOCA_ALLOW_EXPERIMENTAL_API
)

set_target_properties(morpheus_doca
  PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)

if(MORPHEUS_PYTHON_INPLACE_BUILD)
  morpheus_utils_inplace_build_copy(morpheus_doca ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# ##################################################################################################
# - install targets --------------------------------------------------------------------------------

# Get the library directory in a cross-platform way
rapids_cmake_install_lib_dir(lib_dir)

install(
  TARGETS
    morpheus_doca
  EXPORT
    ${PROJECT_NAME}-core-exports
  LIBRARY
    DESTINATION ${lib_dir}
  FILE_SET
    public_headers
)
