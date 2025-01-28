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

list(APPEND CMAKE_MESSAGE_CONTEXT "docs")

find_package(Doxygen REQUIRED dot doxygen)
find_package(Sphinx REQUIRED)

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/source)
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/html)
set(SPHINX_LINKCHECK_OUT ${CMAKE_CURRENT_BINARY_DIR}/linkcheck)
set(SPHINX_ARGS -j auto -T -W)
set(SPHINX_HTML_ARGS -b html ${SPHINX_ARGS})
set(SPHINX_LINKCHECK_ARGS -b linkcheck ${SPHINX_ARGS})

add_custom_target(${PROJECT_NAME}_docs
   COMMAND
      BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR} ${SPHINX_EXECUTABLE} ${SPHINX_HTML_ARGS} ${SPHINX_SOURCE} ${SPHINX_BUILD}
   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
   COMMENT "Generating documentation with Sphinx"
   DEPENDS morpheus-package-outputs morpheus_llm-package-outputs morpheus_dfp-package-outputs
)

add_custom_target(${PROJECT_NAME}_docs_linkcheck
   COMMAND
      BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR} ${SPHINX_EXECUTABLE} ${SPHINX_LINKCHECK_ARGS} ${SPHINX_SOURCE} ${SPHINX_LINKCHECK_OUT}
   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
   COMMENT "Checking documentation links with Sphinx"
   DEPENDS morpheus-package-outputs morpheus_llm-package-outputs morpheus_dfp-package-outputs
)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
