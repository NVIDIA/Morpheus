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
#include "morpheus_llm/llm/input_map.hpp"

#include "morpheus/export.h"
#include "morpheus/types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace morpheus::llm {

/**
 * @brief This class wraps LLMNode and is responsible for node's execution. It also manages mapping of node's
 * inputs to parent and sibling nodes.
 */
class MORPHEUS_EXPORT LLMNodeRunner
{
  public:
    /**
     * @brief Construct a new LLMNodeRunner object
     *
     * @param name
     * @param inputs
     * @param node
     */
    LLMNodeRunner(std::string name, input_mappings_t inputs, std::shared_ptr<LLMNodeBase> node);

    /**
     * @brief Destroy the LLMNodeRunner object
     *
     */
    ~LLMNodeRunner();

    /**
     * @brief Execute node and save outputs to context.
     *
     * @param context context for node's execution
     * @return Task<std::shared_ptr<LLMContext>>
     */
    virtual Task<std::shared_ptr<LLMContext>> execute(std::shared_ptr<LLMContext> context);

    /**
     * @brief Get name of node runner typically same as node name.
     *
     * @return const std::string&
     */
    const std::string& name() const;

    /**
     * @brief Get input mappings for this node.
     *
     * @return const input_mappings_t&
     */
    const input_mappings_t& inputs() const;

    /**
     * @brief Get input names from node's siblings.
     *
     * @return const std::vector<std::string>&
     */
    const std::vector<std::string>& sibling_input_names() const;

    /**
     * @brief Get input names from node's parent.
     *
     * @return const std::vector<std::string>&
     */
    const std::vector<std::string>& parent_input_names() const;

  private:
    std::string m_name;
    input_mappings_t m_inputs;
    std::shared_ptr<LLMNodeBase> m_node;

    std::vector<std::string> m_sibling_input_names;
    std::vector<std::string> m_parent_input_names;
};

}  // namespace morpheus::llm
