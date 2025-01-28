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

set -e +o pipefail
# set -x
# set -v

# Color variables
export b="\033[0;36m"
export g="\033[0;32m"
export r="\033[0;31m"
export e="\033[0;90m"
export y="\033[0;33m"
export x="\033[0m"

export TRITON_IMAGE=${TRITON_IMAGE:-"nvcr.io/nvidia/morpheus/morpheus-tritonserver-models:25.02"}

# TRITON_GRPC_PORT is only used when TRITON_URL is undefined
export TRITON_GRPC_PORT=${TRITON_GRPC_PORT:-"8001"}
export TRITON_URL=${TRITON_URL:-"localhost:${TRITON_GRPC_PORT}"}

export USE_CPP=${USE_CPP:-1}

# RUN OPTIONS
export RUN_PYTORCH=${RUN_PYTORCH:-0}
export RUN_TRITON_ONNX=${RUN_TRITON_ONNX:-1}
export RUN_TRITON_XGB=${RUN_TRITON_XGB:-1}
export RUN_TRITON_TRT=${RUN_TRITON_TRT:-0}
export RUN_TENSORRT=${RUN_TENSORRT:-0}
