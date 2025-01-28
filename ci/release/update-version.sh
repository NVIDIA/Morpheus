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

## Usage
# Either supply full versions:
#    `bash update-version.sh <current_version> <new_version>`
#    Format is YY.MM.PP - no leading 'v' or trailing 'a'
# Or no versions:
#    `bash update-version.sh`

set -e

# If the user has not supplied the versions, determine them from the git tags
if [[ "$#" -ne 2 ]]; then
   echo "No versions were provided. Using last 2 git tags to determined current and next version"

   # Current version comes from the previous alpha tag
   CURRENT_FULL_VERSION=$(git tag --merged HEAD --list 'v*' | sort --version-sort | tail -n 2 | head -n 1 | tr -d 'va')
   # Next version comes from the latest alpha tag
   NEXT_FULL_VERSION=$(git tag --merged HEAD --list 'v*' | sort --version-sort | tail -n 1 | tr -d 'va')
else
   # User has supplied current and next arguments
   CURRENT_FULL_VERSION=$1
   NEXT_FULL_VERSION=$2
fi

CURRENT_MAJOR=$(echo ${CURRENT_FULL_VERSION} | awk '{split($0, a, "."); print a[1]}')
CURRENT_MINOR=$(echo ${CURRENT_FULL_VERSION} | awk '{split($0, a, "."); print a[2]}')
CURRENT_PATCH=$(echo ${CURRENT_FULL_VERSION} | awk '{split($0, a, "."); print a[3]}')
CURRENT_SHORT_TAG=${CURRENT_MAJOR}.${CURRENT_MINOR}

NEXT_MAJOR=$(echo ${NEXT_FULL_VERSION} | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo ${NEXT_FULL_VERSION} | awk '{split($0, a, "."); print a[2]}')
NEXT_PATCH=$(echo ${NEXT_FULL_VERSION} | awk '{split($0, a, "."); print a[3]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}

# Need to distutils-normalize the versions for some use cases
CURRENT_SHORT_TAG_PEP440=$(python -c "from packaging import version; print(version.Version('${CURRENT_SHORT_TAG}'))")
NEXT_SHORT_TAG_PEP440=$(python -c "from packaging import version; print(version.Version('${NEXT_SHORT_TAG}'))")

echo "Preparing release $CURRENT_FULL_VERSION (PEP ${CURRENT_SHORT_TAG_PEP440}) => $NEXT_FULL_VERSION (PEP ${NEXT_SHORT_TAG_PEP440})"

# Inplace sed replace; workaround for Linux and Mac. Accepts multiple files
function sed_runner() {

   pattern=$1
   shift

   for f in $@ ; do
      sed -i.bak ''"$pattern"'' "$f" && rm -f "$f.bak"
   done
}

# .gitmodules
git submodule set-branch -b branch-${NEXT_SHORT_TAG} external/morpheus-visualizations
git submodule set-branch -b branch-${NEXT_SHORT_TAG} external/utilities

if [[ "$(git diff --name-only | grep .gitmodules)" != "" ]]; then
   # Only update the submodules if setting the branch changed .gitmodules. Otherwise this will undo the current commit
   # for any submodules resulting in differences always appearing in CI
   git submodule update --remote --recursive
fi

# Root CMakeLists.txt
sed_runner 's/'"VERSION ${CURRENT_FULL_VERSION}.*"'/'"VERSION ${NEXT_FULL_VERSION}"'/g' CMakeLists.txt

# Root manifest.yaml
sed_runner "s|branch-${CURRENT_SHORT_TAG}|branch-${NEXT_SHORT_TAG}|g" manifest.yaml

# Depedencies file
sed_runner "s/mrc=${CURRENT_SHORT_TAG}/mrc=${NEXT_SHORT_TAG}/g" dependencies.yaml
sed_runner "s/morpheus-dfp=${CURRENT_SHORT_TAG}/morpheus-dfp=${NEXT_SHORT_TAG}/g" dependencies.yaml

# Generate the environment files based upon the updated dependencies.yaml
rapids-dependency-file-generator

# examples/digital_fingerprinting
sed_runner "s/v${CURRENT_FULL_VERSION}-runtime/v${NEXT_FULL_VERSION}-runtime/g" \
   examples/digital_fingerprinting/production/docker-compose.yml \
   examples/digital_fingerprinting/production/Dockerfile
sed_runner "s/v${CURRENT_FULL_VERSION}-runtime/v${NEXT_FULL_VERSION}-runtime/g" examples/digital_fingerprinting/production/Dockerfile

# examples/developer_guide
sed_runner 's/'"VERSION ${CURRENT_FULL_VERSION}.*"'/'"VERSION ${NEXT_FULL_VERSION}"'/g' \
   examples/developer_guide/3_simple_cpp_stage/CMakeLists.txt \
   examples/developer_guide/4_rabbitmq_cpp_stage/CMakeLists.txt

# docs/source/basics/overview.rst
sed_runner "s|blob/branch-${CURRENT_SHORT_TAG}|blob/branch-${NEXT_SHORT_TAG}|g" docs/source/basics/overview.rst

# docs/source/cloud_deployment_guide.md
sed_runner "s|${CURRENT_SHORT_TAG}.tgz|${NEXT_SHORT_TAG}.tgz|g" docs/source/cloud_deployment_guide.md
sed_runner "s|blob/branch-${CURRENT_SHORT_TAG}|blob/branch-${NEXT_SHORT_TAG}|g" docs/source/cloud_deployment_guide.md
sed_runner "s|tree/branch-${CURRENT_SHORT_TAG}|tree/branch-${NEXT_SHORT_TAG}|g" docs/source/cloud_deployment_guide.md

# docs/source/developer_guide/guides/5_digital_fingerprinting.md
sed_runner "s|blob/branch-${CURRENT_SHORT_TAG}|blob/branch-${NEXT_SHORT_TAG}|g" docs/source/developer_guide/guides/5_digital_fingerprinting.md

# docs/source/examples.md
sed_runner "s|blob/branch-${CURRENT_SHORT_TAG}|blob/branch-${NEXT_SHORT_TAG}|g" docs/source/examples.md

# docs/source/getting_started.md
# Only do the minor version here since the full version can mess up the examples
sed_runner "s/${CURRENT_SHORT_TAG}/${NEXT_SHORT_TAG}/g" docs/source/getting_started.md

# models/model-cards
sed_runner "s|blob/branch-${CURRENT_SHORT_TAG}|blob/branch-${NEXT_SHORT_TAG}|g" models/model-cards/*.md
sed_runner "s|tree/branch-${CURRENT_SHORT_TAG}|tree/branch-${NEXT_SHORT_TAG}|g" models/model-cards/*.md

# thirdparty
sed_runner "s|tree/branch-${CURRENT_SHORT_TAG}|tree/branch-${NEXT_SHORT_TAG}|g" thirdparty/README.md

# Update the version of the Morpheus model container
# We need to update several files, however we need to avoid symlinks as well as the build and .cache directories
DOCS_MD_FILES=$(find -P ./docs/source/ -type f -iname "*.md")
EXAMPLES_MD_FILES=$(find -P ./examples/ -type f -iname "*.md")
sed_runner "s|morpheus-tritonserver-models:${CURRENT_SHORT_TAG}|morpheus-tritonserver-models:${NEXT_SHORT_TAG}|g" \
   ${DOCS_MD_FILES} \
   ${EXAMPLES_MD_FILES} \
   .devcontainer/docker-compose.yml \
   examples/sid_visualization/docker-compose.yml \
   models/triton-model-repo/README.md \
   scripts/validation/val-globals.sh \
   tests/benchmarks/README.md
