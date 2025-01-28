#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

function(find_and_configure_rabbitmq version)

  list(APPEND CMAKE_MESSAGE_CONTEXT "rabbitmq")

  # Commit 7fa7b0b contains unreleased cmake fixes which currently only exist in the master branch of the repo.
  # https://github.com/alanxz/rabbitmq-c/issues/740

  rapids_cpm_find(rabbitmq ${version}
    GLOBAL_TARGETS
      rabbitmq rabbitmq::rabbitmq
    BUILD_EXPORT_SET
      ${PROJECT_NAME}-core-exports
    INSTALL_EXPORT_SET
      ${PROJECT_NAME}-core-exports
    CPM_ARGS
      GIT_REPOSITORY  https://github.com/alanxz/rabbitmq-c
      GIT_SHALLOW     TRUE
      OPTIONS         "BUILD_EXAMPLES OFF"
                      "BUILD_TESTING OFF"
                      "BUILD_TOOLS OFF"
  )

  if(rabbitmq_ADDED)
    set(rabbitmq_SOURCE_DIR "${rabbitmq_SOURCE_DIR}" PARENT_SCOPE)
    set(rabbitmq_BINARY_DIR "${rabbitmq_BINARY_DIR}" PARENT_SCOPE)
  endif()

endfunction()

find_and_configure_rabbitmq(${RABBITMQ_VERSION})
