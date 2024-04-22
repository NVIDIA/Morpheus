/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/llm/utils.hpp"

#include "morpheus/llm/input_map.hpp"
#include "morpheus/utilities/string_util.hpp"

#include <glog/logging.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace morpheus::llm {

// Doxygen has problems parsing this regex
#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
// Use this regex
const std::regex VALID_INPUT_NAME(R"([a-zA-Z_][a-zA-Z0-9_]*)", std::regex_constants::ECMAScript);
#endif  // DOXYGEN_SHOULD_SKIP_THIS

bool is_valid_node_name(std::string_view name)
{
    return std::regex_match(name.begin(), name.end(), VALID_INPUT_NAME);
}

bool find_matching_input_for_placeholder(UserInputMapping& input_map,
                                         size_t curr_idx,
                                         const std::vector<std::string>& input_names)
{
    CHECK_EQ(input_map.internal_name, "-")
        << "Called find_matching_input_for_placeholder() with a non placeholder input";

    std::string raw_name = input_map.external_name;

    // If we start with a slash, that means we are mapping from another node, not a parent. Try to find a matching name
    // from the last name. i.e. /node/input -> input
    if (raw_name[0] == '/')
    {
        // Decompose it
        auto name_j_pointer = nlohmann::json::json_pointer(raw_name);

        raw_name = name_j_pointer.back();
    }

    // Match by name
    auto found = std::find(input_names.begin(), input_names.end(), raw_name);

    if (found != input_names.end())
    {
        input_map.internal_name = *found;
        return true;
    }

    if (curr_idx >= input_names.size())
    {
        throw std::invalid_argument(MORPHEUS_CONCAT_STR(
            "Invalid input name '"
            << input_map.external_name
            << "'. Unable to automatically map the external node to an internal name. Matching by name failed and "
               "current index exceeds the bounds of the input names. Current index: "
            << curr_idx << ", Input names size: " << input_names.size()));
    }

    input_map.internal_name = input_names[curr_idx];
    return false;
}

input_mappings_t process_input_names(user_input_mappings_t user_inputs, const std::vector<std::string>& input_names)
{
    input_mappings_t intermediate_inputs;
    input_mappings_t final_inputs;
    user_input_mappings_t wildcard_inputs;

    // The process for converting user specified inputs into the final inputs is as follows:
    // 1. Loop over all inputs and replace any placeholder inputs with the actual inputs
    //    a. If the node name is "-", then replace it with the input name
    //    b. If the node name contains "*", then separate it out to process wildcards last
    // 2. Loop over all wildcard inputs and replace wildcards with remaining input names

    bool is_matching_by_name = false;

    // Loop over all inputs replacing '-' placeholders and separating out wildcards
    for (size_t i = 0; i < user_inputs.size(); ++i)
    {
        auto& single_input = user_inputs[i];

        bool found_star_node_name  = single_input.external_name.find('*') != std::string::npos;
        bool found_star_input_name = single_input.internal_name == "*";

        if (found_star_input_name != found_star_node_name)
        {
            if (found_star_input_name)
            {
                throw std::invalid_argument(
                    "LLMNode::add_node() called with a placeholder external name but no placeholder internal name");
            }
            else
            {
                throw std::invalid_argument(
                    "LLMNode::add_node() called with a placeholder internal name but no placeholder external name");
            }
        }
        else if (found_star_input_name && found_star_node_name)
        {
            // Need to process these after the non-placeholder inputs
            wildcard_inputs.push_back(single_input);
        }
        else
        {
            // No placeholder, so just add the input. If the node_name == "-", then replace it with the input name
            if (single_input.internal_name == "-")
            {
                // We have a placeholder input name, so we need to find the matching input name
                bool matched_by_name = find_matching_input_for_placeholder(single_input, i, input_names);

                if (matched_by_name)
                {
                    is_matching_by_name = true;
                }
                else if (is_matching_by_name)
                {
                    throw std::invalid_argument(MORPHEUS_CONCAT_STR(
                        "Invalid input name '" << single_input.external_name
                                               << "'. Unable to automatically map the external node to an internal "
                                                  "name. Cannot mix matching by name and matching by index"));
                }
            }

            // Add it to the final list
            final_inputs.emplace_back(single_input.external_name, single_input.internal_name);
        }
    }

    // Finally, process the wildcards
    if (!wildcard_inputs.empty())
    {
        // TODO(MDD): Support multiple placeholders
        CHECK_EQ(wildcard_inputs.size(), 1) << "Only a single placeholder input is currently supported";

        std::set<std::string> specified_names;

        std::transform(final_inputs.begin(),
                       final_inputs.end(),
                       std::inserter(specified_names, specified_names.begin()),
                       [](const auto& input) {
                           return input.internal_name;
                       });

        std::set<std::string> total_names(input_names.begin(), input_names.end());

        std::vector<std::string> remaining_names;

        // Find the remaining names
        std::set_difference(total_names.begin(),
                            total_names.end(),
                            specified_names.begin(),
                            specified_names.end(),
                            std::back_inserter(remaining_names));

        auto star_input_name_loc = wildcard_inputs[0].external_name.find('*');

        // Loop over the remaining names and add them to the final inputs
        for (const auto& remaining_name : remaining_names)
        {
            // Make a copy of the string to avoid modifying the original
            auto replaced = std::string(wildcard_inputs[0].external_name);
            replaced.replace(star_input_name_loc, 1, remaining_name);
            final_inputs.emplace_back(replaced, remaining_name);
        }
    }

    if (input_names.size() != final_inputs.size())
    {
        throw std::invalid_argument(MORPHEUS_CONCAT_STR(
            "The number of inputs provided does not match the number of inputs expected by the node. Provided: "
            << final_inputs.size() << ", Expected: " << input_names.size()));
    }

    std::set<std::string> specified_names;

    std::transform(final_inputs.begin(),
                   final_inputs.end(),
                   std::inserter(specified_names, specified_names.begin()),
                   [](const auto& input) {
                       return input.internal_name;
                   });

    std::set<std::string> total_names(input_names.begin(), input_names.end());

    if (specified_names != total_names)
    {
        throw std::invalid_argument(MORPHEUS_CONCAT_STR(
            "The names of the inputs provided do not match the names of the inputs expected by the node. Provided: "
            << StringUtil::array_to_str(specified_names.begin(), specified_names.end())
            << ", Expected: " << StringUtil::array_to_str(total_names.begin(), total_names.end())));
    }

    return final_inputs;
}

}  // namespace morpheus::llm
