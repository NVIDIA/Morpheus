#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Run the update version script
UPDATE_VERSION_OUTPUT=`${MORPHEUS_ROOT}/ci/release/update-version.sh`

# If any diffs were found, the versions were out of date
VERSIONS_OUT_OF_DATE=$(git diff --name-only)

if [[ "${VERSIONS_OUT_OF_DATE}" != "" ]]; then
   echo -e "\n\n>>>> FAILED: version check; begin output\n\n"
   echo -e "${UPDATE_VERSION_OUTPUT}"
   echo -e "\n\nThe following files have out of date versions:"
   echo -e "${VERSIONS_OUT_OF_DATE}"
   echo -e "\n\n>>>> FAILED: version check; end output\n\n" \
           "To update the versions, run the following and commit the results:\n" \
           "   ./ci/release/update-version.sh\n\n"
   exit 1
fi
