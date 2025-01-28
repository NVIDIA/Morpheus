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

#include "morpheus_llm/llm/llm_node_runner.hpp"

#include "morpheus_llm/llm/llm_context.hpp"
#include "morpheus_llm/llm/llm_node_base.hpp"
#include "morpheus_llm/llm/utils.hpp"

#include "morpheus/utilities/string_util.hpp"

#include <mrc/core/utils.hpp>
#include <mrc/coroutines/task.hpp>  // IWYU pragma: keep
#include <nlohmann/json.hpp>

#include <algorithm>  // for max (implied usage)
#include <coroutine>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace morpheus::llm {

LLMNodeRunner::LLMNodeRunner(std::string name, input_mappings_t inputs, std::shared_ptr<LLMNodeBase> node) :
  m_name(std::move(name)),
  m_inputs(std::move(inputs)),
  m_node(std::move(node))
{
    // Get the inputs of the current node
    auto input_names = m_node->get_input_names();

    // Make sure all names are valid names
    for (const auto& name : input_names)
    {
        if (!is_valid_node_name(name))
        {
            throw std::invalid_argument(MORPHEUS_CONCAT_STR(
                "Invalid node name '" << name << "' in node '" << m_name
                                      << "'. Must be a valid identifier matching the regex `[a-zA-Z_][a-zA-Z0-9_]*`"));
        }
    }

    std::set<std::string> input_names_set(input_names.begin(), input_names.end());
    std::set<std::string> specified_input_names_set;

    // Check all of the external node inputs to make sure they are valid
    for (auto& inp : m_inputs)
    {
        const auto& external_name = inp.external_name;

        if (specified_input_names_set.contains(inp.internal_name))
        {
            throw std::runtime_error("Input '" + inp.internal_name + "' is specified more than once");
        }

        // Determine if the inputs are coming from a parent node or a sibling node
        if (inp.external_name[0] == '/')
        {
            // Break it down into json pointer components
            auto components = nlohmann::json::json_pointer(external_name);

            // Loop over each segment to make sure its a valid name
            while (!components.empty())
            {
                if (!is_valid_node_name(components.back()))
                {
                    throw std::invalid_argument(MORPHEUS_CONCAT_STR(
                        "Invalid component '" << components.back()
                                              << "' in external node mapping name '" + external_name +
                                                     "'. All components must be valid identifiers"));
                }

                components.pop_back();
            }

            m_sibling_input_names.push_back(inp.external_name);
        }
        else
        {
            // Must be a valid identifier
            if (!is_valid_node_name(external_name))
            {
                throw std::invalid_argument(MORPHEUS_CONCAT_STR("Invalid external node mapping name '"
                                                                << external_name
                                                                << "'. Must be a valid identifier matching the regex "
                                                                   "`[a-zA-Z_][a-zA-Z0-9_]*`"));
            }

            m_parent_input_names.push_back(inp.external_name);
        }

        specified_input_names_set.insert(inp.internal_name);
    }

    auto [missing_names, extra_names] = mrc::set_compare(input_names_set, specified_input_names_set);

    if (!missing_names.empty() || !extra_names.empty())
    {
        throw std::runtime_error(
            MORPHEUS_CONCAT_STR("Invalid inputs for node '"
                                << m_name << "'. Contained extra or missing inputs. Extra inputs: "
                                << StringUtil::array_to_str(extra_names.begin(), extra_names.end())
                                << ". Missing inputs: "
                                << StringUtil::array_to_str(missing_names.begin(), missing_names.end()) << "."));
    }
}

LLMNodeRunner::~LLMNodeRunner() = default;

Task<std::shared_ptr<LLMContext>> LLMNodeRunner::execute(std::shared_ptr<LLMContext> context)
{
    // Create a new context
    auto child_context = context->push(m_name, m_inputs);

    // Also need error handling here
    auto returned_context = co_await m_node->execute(child_context);

    // Call pop to apply the outputs to the parent context
    child_context->pop();

    co_return returned_context;
}

const std::string& LLMNodeRunner::name() const
{
    return m_name;
}

const input_mappings_t& LLMNodeRunner::inputs() const
{
    return m_inputs;
}

const std::vector<std::string>& LLMNodeRunner::sibling_input_names() const
{
    return m_sibling_input_names;
}

const std::vector<std::string>& LLMNodeRunner::parent_input_names() const
{
    return m_parent_input_names;
}

}  // namespace morpheus::llm
