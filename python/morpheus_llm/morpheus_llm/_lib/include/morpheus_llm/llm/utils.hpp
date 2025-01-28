/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "morpheus_llm/llm/input_map.hpp"

#include "morpheus/export.h"

#include <string>
#include <string_view>
#include <vector>

namespace morpheus::llm {

/**
 * @brief Determines if a name is valid for a node or input. Must follow python rules for identifiers.
 *
 * @param name The node name to test
 * @return true
 * @return false
 */
bool MORPHEUS_EXPORT is_valid_node_name(std::string_view name);

/**
 * @brief Resolves user input mappings to final mappings. Handles replacement of placeholders in inputs with
 * actual internal input names provided in input_names.
 *
 * @param user_inputs user input mappings
 * @param input_names internal input names for context
 * @return input_mappings_t
 */
input_mappings_t MORPHEUS_EXPORT process_input_names(user_input_mappings_t user_inputs,
                                                     const std::vector<std::string>& input_names);

}  // namespace morpheus::llm
