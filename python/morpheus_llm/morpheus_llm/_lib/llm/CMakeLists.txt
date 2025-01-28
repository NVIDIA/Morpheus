# =============================================================================
# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

list(APPEND CMAKE_MESSAGE_CONTEXT "llm")

set(MODULE_SOURCE_FILES
   src/py_llm_engine.cpp
   src/py_llm_lambda_node.cpp
   src/py_llm_node_base.cpp
   src/py_llm_node.cpp
   src/py_llm_task_handler.cpp
)

# Add the module file
list(APPEND MODULE_SOURCE_FILES module.cpp)

# Create the python module
morpheus_add_pybind11_module(llm
   INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
   SOURCE_FILES ${MODULE_SOURCE_FILES}
   LINK_TARGETS morpheus_llm
   CURRENT_DIR_IS_MODULE
)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
