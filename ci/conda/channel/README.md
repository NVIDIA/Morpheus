<!--
SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

Creates a local Conda channel using Docker Compose and nginx. Can be helpful when testing new Conda packages

To Use:
1. Ensure Docker Compose is installed
2. Set the location of the `conda-bld` folder to host as a Conda channel to the variable `$CONDA_REPO_DIR`
   1. For example, `export CONDA_REPO_DIR=$CONDA_PREFIX/conda-bld`
3. Launch Docker Compose
   1. `docker compose up -d`
4. Install Conda packages using the local channel
   1. `conda install -c http://localhost:8080 <my_package>`
