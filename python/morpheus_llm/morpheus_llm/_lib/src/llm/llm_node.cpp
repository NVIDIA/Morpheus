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

#include "morpheus_llm/llm/llm_node.hpp"

#include "morpheus_llm/llm/llm_context.hpp"
#include "morpheus_llm/llm/llm_node_runner.hpp"
#include "morpheus_llm/llm/utils.hpp"

#include "morpheus/utilities/string_util.hpp"

#include <mrc/coroutines/task.hpp>  // IWYU pragma: keep

#include <algorithm>
#include <coroutine>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace morpheus::llm {

LLMNode::LLMNode() = default;

LLMNode::~LLMNode() = default;

std::shared_ptr<LLMNodeRunner> LLMNode::add_node(std::string name,
                                                 user_input_mappings_t inputs,
                                                 std::shared_ptr<LLMNodeBase> node,
                                                 bool is_output)
{
    // Check if a node with this name already exists
    if (std::find_if(m_child_runners.begin(), m_child_runners.end(), [&name](const auto& runner) {
            return runner->name() == name;
        }) != m_child_runners.end())
    {
        throw std::invalid_argument("A node with the name '" + name + "' already exists");
    }

    // Get the inputs of the current node
    auto input_names = node->get_input_names();

    auto final_inputs = process_input_names(inputs, input_names);

    // Check the final inputs to ensure they match existing nodes
    for (const auto& inp : final_inputs)
    {
        // Find the first occurance of "/"
        auto slash_pos = inp.external_name.find('/');

        if (slash_pos != 0)
        {
            // If there is no slash, then the input is a parent of the current node
            continue;
        }

        // Otherwise, the input is a child of the current node. Find the next slash
        slash_pos = inp.external_name.find('/', 1);

        auto upstream_node_name =
            inp.external_name.substr(1, slash_pos != std::string::npos ? slash_pos - 1 : std::string::npos);

        if (std::find_if(m_child_runners.begin(), m_child_runners.end(), [&upstream_node_name](const auto& runner) {
                return runner->name() == upstream_node_name;
            }) == m_child_runners.end())
        {
            // Could not find a matching node for this input
            throw std::invalid_argument(MORPHEUS_CONCAT_STR(
                "Could not find a node with the name '" << upstream_node_name << "' for the input {'"
                                                        << inp.external_name << "', '" << inp.internal_name << "'}"));
        }
    }

    auto node_runner = std::make_shared<LLMNodeRunner>(std::move(name), std::move(final_inputs), std::move(node));

    // Add the child inputs to the current inputs
    for (const auto& parent_input : node_runner->parent_input_names())
    {
        if (std::find(m_input_names.begin(), m_input_names.end(), parent_input) == m_input_names.end())
        {
            m_input_names.push_back(parent_input);
        }
    }

    // Perform checks that the existing nodes meet the requirements

    m_child_runners.push_back(node_runner);

    if (is_output)
    {
        m_output_node_names.push_back(node_runner->name());
    }

    return node_runner;
}

std::vector<std::string> LLMNode::get_input_names() const
{
    return m_input_names;
}

const std::vector<std::string>& LLMNode::get_output_node_names() const
{
    return m_output_node_names;
}

size_t LLMNode::node_count() const
{
    return m_child_runners.size();
}

Task<std::shared_ptr<LLMContext>> LLMNode::execute(std::shared_ptr<LLMContext> context)
{
    for (auto& runner : m_child_runners)
    {
        // Run the child node
        co_await runner->execute(context);

        // Wait for the child node outputs (This will yield if not already available)
        // context->get_outputs();
    }

    // Before returning, set the output names to only propagate the specified outputs
    context->set_output_names(m_output_node_names);

    co_return context;
}

}  // namespace morpheus::llm
