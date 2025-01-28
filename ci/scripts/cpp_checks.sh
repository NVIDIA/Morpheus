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

# set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/common.sh

IWYU_TOOL=$(find_iwyu_tool)

if [[ "${SKIP_IWYU}" == "" && -z "${IWYU_TOOL}" ]]; then
   echo -e "Could not find Include What You Use tool (using 'which iwyu_tool.py'). Skipping IWYU check. Ensure IWYU is installed and available on PATH\n"
   SKIP_IWYU=1
fi

# Pre-populate the return values in case they are skipped
PRAGMA_CHECK_RETVAL=0
CLANG_TIDY_RETVAL=0
CLANG_FORMAT_RETVAL=0
IWYU_RETVAL=0

# Get the list of modified files inside the trtlab/MORPHEUS folder
get_modified_files ${CPP_FILE_REGEX} MORPHEUS_MODIFIED_FILES

# If there are any files, then run clang tidy
if [[ -n "${MORPHEUS_MODIFIED_FILES}" ]]; then
   echo -e "Running C++ checks on ${#MORPHEUS_MODIFIED_FILES[@]} files:"

   for f in "${MORPHEUS_MODIFIED_FILES[@]}"; do
      echo "  $f"
   done

   # Check for `#pragma once` in any .hpp or .h file
   if [[ "${SKIP_PRAGMA_CHECK}" == "" ]]; then

      PRAGMA_CHECK_OUTPUT=`grep -rL --include=\*.{h,hpp} --exclude-dir={build,.cache} -e '#pragma once' $(get_modified_files $CPP_FILE_REGEX)`

      if [[ -n "${PRAGMA_CHECK_OUTPUT}" ]]; then
         PRAGMA_CHECK_RETVAL=1
      fi
   fi

   # Clang-tidy
   if [[ "${SKIP_CLANG_TIDY}" == "" ]]; then

      CLANG_TIDY_DIFF=$(find_clang_tidy_diff)

      # Run using a clang-tidy wrapper to allow warnings-as-errors and to eliminate any output except errors (since
      # clang-tidy-diff.py doesn't return the correct error codes)
      CLANG_TIDY_OUTPUT=`get_unified_diff ${CPP_FILE_REGEX} | ${CLANG_TIDY_DIFF} \
         -extra-arg="-Wno-ignored-optimization-argument" -j 0 -path ${BUILD_DIR} -p1 -quiet 2>&1`

      if [[ -n "${CLANG_TIDY_OUTPUT}" && ${CLANG_TIDY_OUTPUT} != "No relevant changes found." ]]; then
         CLANG_TIDY_RETVAL=1
      fi
   fi

   # Clang-format
   if [[ "${SKIP_CLANG_FORMAT}" == "" ]]; then

      CLANG_FORMAT_DIFF=$(find_clang_format_diff)

      CLANG_FORMAT_OUTPUT=`get_unified_diff ${CPP_FILE_REGEX} | ${CLANG_FORMAT_DIFF} -p1 2>&1`

      if [[ -n "${CLANG_FORMAT_OUTPUT}" ]]; then
         CLANG_FORMAT_RETVAL=1
      fi
   fi

   # Include What You Use
   if [[ "${SKIP_IWYU}" == "" ]]; then
      # Remove .h, .hpp, and .cu files from the modified list
      shopt -s extglob
      IWYU_MODIFIED_FILES=( "${MORPHEUS_MODIFIED_FILES[@]/*.@(h|hpp|cu)/}" )

      # Get the list of compiled files relative to this directory
      WORKING_PREFIX="${PWD}/"
      COMPILED_FILES=( $(jq -r .[].file ${BUILD_DIR}/compile_commands.json | sort -u ) )
      COMPILED_FILES=( "${COMPILED_FILES[@]/#$WORKING_PREFIX/}" )
      COMBINED_FILES=("${COMPILED_FILES[@]}")
      COMBINED_FILES+=("${IWYU_MODIFIED_FILES[@]}")

      # Find the intersection between compiled files and modified files
      IWYU_MODIFIED_FILES=( $(printf '%s\0' "${COMBINED_FILES[@]}" | sort -z | uniq -d -z | xargs -0n1) )

      if [[ "${#IWYU_MODIFIED_FILES[@]}" != "0" ]]; then
         NUM_PROC=$(get_num_proc)
         IWYU_OUTPUT=`${IWYU_TOOL} -p ${BUILD_DIR} -j ${NUM_PROC} ${IWYU_MODIFIED_FILES[@]} 2>&1`
         IWYU_RETVAL=$?
      else
         echo "No modified C++ files match IWYU's filters. Skipping IWYU"

         # Set the return value to 0 if we didnt run it
         IWYU_RETVAL=0
      fi
   fi
else
   echo "No modified C++ files to check"
fi

# Now check the values
if [[ "${SKIP_PRAGMA_CHECK}" != "" ]]; then
   echo -e "\n\n>>>> SKIPPED: pragma check\n\n"
elif [[ "${PRAGMA_CHECK_RETVAL}" != "0" ]]; then
   echo -e "\n\n>>>> FAILED: pragma check; begin output\n\n"
   echo -e "Missing \`#pragma once\` in the following files:"
   echo -e "${PRAGMA_CHECK_OUTPUT}"
   echo -e "\n\n>>>> FAILED: pragma check; end output\n\n" \
           "To auto-fix many issues (not all) run:\n" \
           "   ./ci/scripts/fix_all.sh\n\n"
else
   echo -e "\n\n>>>> PASSED: pragma check\n\n"
fi

if [[ "${SKIP_CLANG_TIDY}" != "" ]]; then
   echo -e "\n\n>>>> SKIPPED: clang-tidy check\n\n"
elif [[ "${CLANG_TIDY_RETVAL}" != "0" ]]; then
   echo -e "\n\n>>>> FAILED: clang-tidy check; begin output\n\n"
   echo -e "${CLANG_TIDY_OUTPUT}"
   echo -e "\n\n>>>> FAILED: clang-tidy check; end output\n\n" \
           "To auto-fix many issues (not all) run:\n" \
           "   ./ci/scripts/fix_all.sh\n\n"
else
   echo -e "\n\n>>>> PASSED: clang-tidy check\n\n"
fi

if [[ "${SKIP_CLANG_FORMAT}" != "" ]]; then
   echo -e "\n\n>>>> SKIPPED: clang-format check\n\n"
elif [[ "${CLANG_FORMAT_RETVAL}" != "0" ]]; then
   echo -e "\n\n>>>> FAILED: clang-format check; begin output\n\n"
   echo -e "${CLANG_FORMAT_OUTPUT}"
   echo -e "\n\n>>>> FAILED: clang-format check; end output\n\n" \
           "To auto-fix many issues (not all) run:\n" \
           "   ./ci/scripts/fix_all.sh\n\n"
else
   echo -e "\n\n>>>> PASSED: clang-format check\n\n"
fi

if [[ "${SKIP_IWYU}" != "" ]]; then
   echo -e "\n\n>>>> SKIPPED: include-what-you-use check\n\n"
elif [[ "${IWYU_RETVAL}" != "0" ]]; then
   echo -e "\n\n>>>> FAILED: include-what-you-use check; begin output\n\n"
   echo -e "${IWYU_OUTPUT}"
   echo -e "\n\n>>>> FAILED: include-what-you-use check; end output\n\n" \
           "To auto-fix many issues (not all) run:\n" \
           "   ./ci/scripts/fix_all.sh\n\n"
else
   echo -e "\n\n>>>> PASSED: include-what-you-use check\n\n"
fi

RETVALS=(${CLANG_TIDY_RETVAL} ${CLANG_FORMAT_RETVAL} ${IWYU_RETVAL})
IFS=$'\n'
RETVAL=`echo "${RETVALS[*]}" | sort -nr | head -n1`

exit $RETVAL
