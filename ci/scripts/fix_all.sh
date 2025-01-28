#!/bin/bash
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


# This will run all style cleanup tools on the source code
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/../scripts/common.sh

# If IGNORE_GIT_DIFF is enabled, use all files
if [[ "${IGNORE_GIT_DIFF}" == "1" ]]; then
   PY_MODIFIED_FILES=$(find ./ -name '*' | grep -P "${PYTHON_FILE_REGEX}")
   CPP_MODIFIED_FILES=$(find ./ -name '*' | grep -P "${CPP_FILE_REGEX}")
else
   # Get the list of modified files to check
   get_modified_files ${PYTHON_FILE_REGEX} PY_MODIFIED_FILES
   get_modified_files ${CPP_FILE_REGEX} CPP_MODIFIED_FILES
fi

# Run copyright fix
if [[ "${SKIP_COPYRIGHT}" == "" ]]; then
   echo "Running copyright check..."

   # If IGNORE_GIT_DIFF is enabled, use all files
   if [[ "${IGNORE_GIT_DIFF}" == "1" ]]; then
      python3 ./ci/scripts/copyright.py --fix-all ./ 2>&1
   else
      python3 ./ci/scripts/copyright.py --fix-all --git-modified-only ./ 2>&1
   fi
fi

# Run clang-tidy
if [[ "${SKIP_CLANG_TIDY}" == "" ]]; then

   # If IGNORE_GIT_DIFF is enabled, use all files
   if [[ "${IGNORE_GIT_DIFF}" == "1" ]]; then
      # Now find clang-tidy
      export CLANG_TIDY=$(find_clang_tidy)

      echo "Running clang-tidy from '${CLANG_TIDY}'..."
      run-clang-tidy -clang-tidy-binary ${SCRIPT_DIR}/run_clang_tidy_for_ci.sh -j 0 -p ${BUILD_DIR} -fix -quiet ${CPP_MODIFIED_FILES[@]} 2>&1
   else
      CLANG_TIDY_DIFF=$(find_clang_tidy_diff)

      # Use -n here since the output could be multiple commands
      if [[ -n "${CLANG_TIDY_DIFF}" ]]; then
         echo "Running clang-tidy from '${CLANG_TIDY_DIFF}'..."
         get_unified_diff ${CPP_FILE_REGEX} | ${CLANG_TIDY_DIFF} -p1 -j 0 -path ${BUILD_DIR} -fix -quiet 2>&1
      else
         echo "Skipping clang-tidy. Could not find clang-tidy-diff.py at '${CLANG_TIDY_DIFF}'"
      fi
   fi
fi

# Run include-what-you-use
if [[ "${SKIP_IWYU}" == "" && "${CPP_MODIFIED_FILES}" != "" ]]; then

   IWYU_TOOL=$(find_iwyu_tool)

   if [[ -x "${IWYU_TOOL}" ]]; then
      echo "Running include-what-you-use from '${IWYU_TOOL}'..."
      ${IWYU_TOOL} -j $(nproc) -p ${BUILD_DIR} morpheus | fix_includes.py --nosafe_headers --nocomments 2>&1
   else
      echo "Skipping include-what-you-use. Could not find iwyu_tool.py at '${IWYU_TOOL}'"
   fi
fi

# Run clang-format (Should be after IWYU because that messes up include ordering)
if [[ "${SKIP_CLANG_FORMAT}" == "" ]]; then

   # If IGNORE_GIT_DIFF is enabled, use all files
   if [[ "${IGNORE_GIT_DIFF}" == "1" ]]; then
      echo "Running clang-format from '${SCRIPT_DIR}/run_clang_format.py'..."
      python3 ${SCRIPT_DIR}/run_clang_format.py -inplace -regex "${CPP_FILE_REGEX}" ./ 2>&1
   else
      CLANG_FORMAT_DIFF=$(find_clang_format_diff)

      if [[ -x "${CLANG_FORMAT_DIFF}" ]]; then
         echo "Running clang-format from '${CLANG_FORMAT_DIFF}'..."
         get_unified_diff ${CPP_FILE_REGEX} | ${CLANG_FORMAT_DIFF} -p1 -i -sort-includes 2>&1
      else
         echo "Skipping clang-format. Could not find clang-format-diff at '${CLANG_FORMAT_DIFF}'"
      fi
   fi
fi

# Run isort
if [[ "${SKIP_ISORT}" == "" ]]; then
   echo "Running isort..."
   python3 -m isort --settings-file ${PROJ_TOML} ${PY_MODIFIED_FILES[@]} 2>&1
fi

# Run yapf
if [[ "${SKIP_YAPF}" == "" ]]; then
   echo "Running yapf..."
   python3 -m yapf -i --style ${PROJ_TOML} -r ${PY_MODIFIED_FILES[@]}
fi
