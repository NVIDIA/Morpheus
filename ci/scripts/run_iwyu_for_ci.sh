#!/bin/bash
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

# set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Its possible to fix violations using this script. If any errors are reported, run the following from the repo root:
# ci/scripts/run_iwyu_for_ci.sh -j 6 -p ./build morpheus | fix_includes.py --nosafe_headers --nocomments

# Call iwyu_tool.py and append IWYU arguments onto the end
${IWYU_TOOL_PY:-iwyu_tool.py} "$@" -- \
   -Xiwyu --mapping_file=${MORPHEUS_ROOT:-${SCRIPT_DIR}/../..}/ci/iwyu/mappings.imp \
   -Xiwyu --verbose=${IWYU_VERBOSITY:-1} \
   -Xiwyu --no_fwd_decls \
   -Xiwyu --quoted_includes_first \
   -Xiwyu --cxx17ns \
   -Xiwyu --max_line_length=120 \
   -Xiwyu --error=1 \
   --driver-mode=g++
