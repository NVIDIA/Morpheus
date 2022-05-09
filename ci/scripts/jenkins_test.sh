#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -e

source ci/scripts/jenkins_common.sh

gpuci_logger "Check versions"
python3 --version
gcc --version
g++ --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

gpuci_logger "Downloading build artifacts"
aws s3 cp --no-progress "${ARTIFACT_URL}/conda.tar.gz" "${WORKSPACE_TMP}/conda.tar.gz"
aws s3 cp --no-progress "${ARTIFACT_URL}/build.tar.gz" "${WORKSPACE_TMP}/build.tar.gz"

gpuci_logger "Extracting"
mkdir -p /opt/conda/envs/morpheus
tar xf "${WORKSPACE_TMP}/conda.tar.gz" --directory /opt/conda/envs/morpheus
tar xf "${WORKSPACE_TMP}/build.tar.gz" --directory ./

gpuci_logger "Setting up test deps"
conda activate morpheus
conda-unpack

npm install --silent -g camouflage-server
mamba install -q -y -c conda-forge "git-lfs=3.1.4"

gpuci_logger "Pulling LFS assets"
git lfs install
git lfs pull

gpuci_logger "Running tests"
pytest --run_slow
exit $?
