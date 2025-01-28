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
#include "morpheus/types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace morpheus::llm {

/**
 * @brief Base class for LLMNode.
 */
class MORPHEUS_EXPORT LLMNodeBase
{
  public:
    /**
     * @brief Destroy the LLMNodeBase object.
     */
    virtual ~LLMNodeBase() = default;

    /**
     * @brief Virtual method for implementing how task handler gets its input names.
     *
     * @return std::vector<std::string>
     */
    virtual std::vector<std::string> get_input_names() const = 0;

    /**
     * @brief Virtual method for implementing the execution of a node.
     *
     * @param context context for node's execution
     * @return Task<std::shared_ptr<LLMContext>>
     */
    virtual Task<std::shared_ptr<LLMContext>> execute(std::shared_ptr<LLMContext> context) = 0;
};

}  // namespace morpheus::llm
