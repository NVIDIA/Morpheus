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
# =============================================================================

if(MORPHEUS_USE_IWYU)
  morpheus_utils_initialize_iwyu(
      MORPHEUS_USE_IWYU
      MORPHEUS_IWYU_VERBOSITY
      MORPHEUS_IWYU_PROGRAM
      MORPHEUS_IWYU_OPTIONS
      MORPHEUS_USE_CCACHE
  )
endif(MORPHEUS_USE_IWYU)
