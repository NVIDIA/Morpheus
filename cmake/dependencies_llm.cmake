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

list(APPEND CMAKE_MESSAGE_CONTEXT "dep_llm")

# pybind11
# =========
morpheus_utils_configure_pybind11()

# RD-Kafka
# =====
morpheus_utils_configure_rdkafka()

# MRC (Should come after all third party but before NVIDIA repos)
# =====
morpheus_utils_configure_mrc()

# CuDF
morpheus_utils_configure_cudf()

# Triton-client
# =====
morpheus_utils_configure_tritonclient()

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
