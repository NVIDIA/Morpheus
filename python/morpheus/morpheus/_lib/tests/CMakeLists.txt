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

list(APPEND CMAKE_MESSAGE_CONTEXT "tests")

find_package(pybind11 REQUIRED)

include(GoogleTest)

# Cuda Test
add_executable(test_cuda
  test_cuda.cu
)

target_link_libraries(test_cuda
  PRIVATE
    glog::glog
    GTest::gtest
    GTest::gtest_main
    matx::matx
)

gtest_discover_tests(test_cuda)

set_target_properties(test_cuda
  PROPERTIES
    INSTALL_RPATH "$ORIGIN/.."
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# Morpheus Test Utilities
add_library(
  morpheus_test_utilities
    test_utils/common.cpp
    test_utils/tensor_utils.cpp
)

target_link_libraries(
  morpheus_test_utilities
  PUBLIC
    GTest::gtest
    morpheus
)

# Ensure that the python executable is defined for the tests
target_compile_definitions(morpheus_test_utilities
  PRIVATE
    PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}"
)

# Morpheus Tests
function(add_morpheus_test)
  set(options)
  set(oneValueArgs NAME)
  set(multiValueArgs FILES)
  cmake_parse_arguments(
    MORPHEUS_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN}
  )

  add_executable(test_${MORPHEUS_TEST_NAME}
    ${MORPHEUS_TEST_FILES}
  )

  target_link_libraries(test_${MORPHEUS_TEST_NAME}
    PRIVATE
      GTest::gtest_main
      morpheus_test_utilities
      pybind11::embed
      TritonClient::httpclient_static
  )

  gtest_discover_tests(test_${MORPHEUS_TEST_NAME})

  set_target_properties(test_${MORPHEUS_TEST_NAME}
    PROPERTIES
      INSTALL_RPATH "$ORIGIN/.."
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
  )
endfunction()

add_morpheus_test(
  NAME io
  FILES
    io/test_data_loader.cpp
    io/test_data_loader_registry.cpp
    io/test_loaders.cpp
)

add_morpheus_test(
  NAME messages
  FILES
    messages/test_control_message.cpp
    messages/test_dev_doc_ex3.cpp
    messages/test_sliced_message_meta.cpp
    messages/test_message_meta.cpp
)

add_morpheus_test(
  NAME modules
  FILES
    modules/test_data_loader_module.cpp
)

add_morpheus_test(
  NAME objects
  FILES
    objects/test_dtype.cpp
)

add_morpheus_test(
  NAME deserializers
  FILES
    test_deserializers.cpp
)

add_morpheus_test(
  NAME dev_mem_info
  FILES
    test_dev_mem_info.cpp
)

add_morpheus_test(
  NAME file_in_out
  FILES
    test_file_in_out.cpp
)

add_morpheus_test(
  NAME matx_util
  FILES
    test_matx_util.cpp
)

add_morpheus_test(
  NAME multi_slices
  FILES
    test_multi_slices.cpp
)

add_morpheus_test(
  NAME controllers
  FILES
    controllers/test_monitor_controller.cpp
)

add_morpheus_test(
  NAME stages
  FILES
    stages/test_preprocess_nlp.cpp
    stages/test_preprocess_fil.cpp
    stages/test_add_scores.cpp
    stages/test_add_classification.cpp
)

add_morpheus_test(
  NAME tensor
  FILES
    test_tensor.cpp
)

add_morpheus_test(
  NAME triton_inference_stage
  FILES
    stages/test_triton_inference_stage.cpp
)

add_morpheus_test(
  NAME type_util
  FILES
    test_type_util.cpp
)

add_morpheus_test(
  NAME table_util
  FILES
    utilities/test_table_util.cpp
)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
