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

#include "morpheus_llm/llm/fwd.hpp"

#include "morpheus/export.h"

#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace morpheus::llm {

struct MORPHEUS_EXPORT InputMap
{
    InputMap();

    InputMap(std::string external_name, std::string internal_name);

    std::string external_name;  // The name of the upstream node to use as input
    std::string internal_name;  // The name of the input that the upstream node maps to.
};

/**
 * @brief Represents the options that a user can specify for an input mapping. Will get converted into an InputMap.
 */
struct MORPHEUS_EXPORT UserInputMapping
{
    UserInputMapping();

    UserInputMapping(std::string external_name, std::string internal_name = "-");

    UserInputMapping(const std::shared_ptr<LLMNodeRunner>& runner);

    UserInputMapping(std::tuple<std::string, std::string>&& tuple);

    std::string external_name;  // The name of the upstream node to use as input
    std::string internal_name;  // The name of the input that the upstream node maps to.
};

// Ordered mapping of input names (current node) to output names (from previous nodes)
using input_mappings_t = std::vector<InputMap>;

using user_input_mappings_t = std::vector<UserInputMapping>;

}  // namespace morpheus::llm
