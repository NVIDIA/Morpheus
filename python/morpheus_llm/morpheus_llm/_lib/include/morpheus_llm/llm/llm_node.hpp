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
#include "morpheus_llm/llm/llm_node_base.hpp"

#include "morpheus/export.h"
#include "morpheus/types.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace morpheus::llm {

/**
 * @brief This class is used to implement functionality required in the processing of an LLM
 * service request such as a prompt generator or LLM service client. Nodes are added to an LLMEngine
 * which manages their execution including mapping of each node's input/output to parent and sibling
 * nodes.
 */
class MORPHEUS_EXPORT LLMNode : public LLMNodeBase
{
  public:
    /**
     * @brief Construct a new LLMNode object.
     */
    LLMNode();

    /**
     * @brief Destroy the LLMNode object.
     */
    ~LLMNode() override;

    /**
     * @brief Add child node to this node and validate user input mappings.
     *
     * @param name child node name
     * @param inputs child node input mappings
     * @param node child node object
     * @param is_output true if output node
     * @return std::shared_ptr<LLMNodeRunner>
     */
    virtual std::shared_ptr<LLMNodeRunner> add_node(std::string name,
                                                    user_input_mappings_t inputs,
                                                    std::shared_ptr<LLMNodeBase> node,
                                                    bool is_output = false);

    /**
     * @brief Get the input names for this node.
     *
     * @return std::vector<std::string>
     */
    std::vector<std::string> get_input_names() const override;

    /**
     * @brief Get the names of output child nodes.
     *
     * @return const std::vector<std::string>&
     */
    const std::vector<std::string>& get_output_node_names() const;

    /**
     * @brief Get number of child nodes.
     *
     * @return size_t
     */
    size_t node_count() const;

    /**
     * @brief Execute all child nodes and save output from output node(s) to context.
     *
     * @param context context for node's execution
     * @return Task<std::shared_ptr<LLMContext>>
     */
    Task<std::shared_ptr<LLMContext>> execute(std::shared_ptr<LLMContext> context) override;

  private:
    std::vector<std::shared_ptr<LLMNodeRunner>> m_child_runners;

    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_node_names;  // Names of nodes to be used as the output
};

}  // namespace morpheus::llm
