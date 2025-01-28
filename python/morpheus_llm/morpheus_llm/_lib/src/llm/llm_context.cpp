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

#include "morpheus_llm/llm/llm_context.hpp"

#include "morpheus/messages/control.hpp"       // IWYU pragma: keep
#include "morpheus/utilities/string_util.hpp"  // for MORPHEUS_CONCAT_STR

#include <nlohmann/json.hpp>  // for basic_json, json

#include <algorithm>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace morpheus::llm {

LLMContext::LLMContext() : m_state(std::make_shared<LLMContextState>()) {}

LLMContext::LLMContext(LLMTask task, std::shared_ptr<ControlMessage> message) : LLMContext()
{
    m_state->task    = std::move(task);
    m_state->message = std::move(message);
}

LLMContext::LLMContext(std::shared_ptr<LLMContext> parent, std::string name, input_mappings_t inputs) :
  m_parent(std::move(parent)),
  m_name(std::move(name)),
  m_inputs(std::move(inputs))
{}

LLMContext::~LLMContext() = default;

std::shared_ptr<LLMContext> LLMContext::parent() const
{
    return m_parent;
}

const std::string& LLMContext::name() const
{
    return m_name;
}

const input_mappings_t& LLMContext::input_map() const
{
    return m_inputs;
}

const LLMTask& LLMContext::task() const
{
    if (m_parent)
    {
        return m_parent->task();
    }

    return m_state->task;
}

std::shared_ptr<ControlMessage>& LLMContext::message() const
{
    if (m_parent)
    {
        return m_parent->message();
    }

    return m_state->message;
}

const utilities::json_t& LLMContext::all_outputs() const
{
    return m_outputs;
}

std::string LLMContext::full_name() const
{
    // Determine the full name
    if (m_parent)
    {
        return m_parent->full_name() + "/" + m_name;
    }

    // If we dont have a parent, we are the root context. So return nothing
    return "";
}

std::shared_ptr<LLMContext> LLMContext::push(std::string name, input_mappings_t inputs)
{
    return std::make_shared<LLMContext>(this->shared_from_this(), std::move(name), std::move(inputs));
}

void LLMContext::pop()
{
    // Copy the outputs from the child context to the parent
    if (m_output_names.empty())
    {
        // Use them all by default
        m_parent->set_output(m_name, std::move(m_outputs));
    }
    else if (m_output_names.size() == 1)
    {
        // Treat only a single output as the output
        m_parent->set_output(m_name, std::move(m_outputs[m_output_names[0]]));
    }
    else
    {
        // Build a new json object with only the specified keys
        utilities::json_t new_outputs;

        for (const auto& output_name : m_output_names)
        {
            new_outputs[output_name] = m_outputs[output_name];
        }

        m_parent->set_output(m_name, std::move(new_outputs));
    }
}

const utilities::json_t& LLMContext::get_input() const
{
    if (m_inputs.size() > 1)
    {
        throw std::runtime_error(
            "LLMContext::get_input() called on a context with multiple inputs. Use get_input(input_name) instead.");
    }

    return this->get_input(m_inputs[0].internal_name);
}

const utilities::json_t& LLMContext::get_input(const std::string& node_name) const
{
    if (node_name[0] == '/')
    {
        nlohmann::json::json_pointer node_json_ptr(node_name);

        if (!m_outputs.contains(node_json_ptr))
        {
            throw std::runtime_error(MORPHEUS_CONCAT_STR("Input '" << node_name << "' not found in the output map"));
        }

        // Get the value from a sibling output
        return m_outputs[node_json_ptr];
    }
    else
    {
        // Must be on the parent, so find the mapping between this namespace and the parent
        auto found = std::find_if(m_inputs.begin(), m_inputs.end(), [&node_name](const auto& map_iterator) {
            return map_iterator.internal_name == node_name;
        });

        if (found == m_inputs.end())
        {
            std::stringstream error_msg;
            error_msg << "Input '" << node_name << "' not found in the input list.";

            if (!m_inputs.empty())
            {
                error_msg << " Available inputs are:";
                for (const auto& input : m_inputs)
                {
                    error_msg << " '" << input.internal_name << "'";
                }
            }
            else
            {
                error_msg << " Input list is empty.";
            }

            throw std::runtime_error(error_msg.str());
        }

        auto& input_name = found->external_name;

        // Get the value from a parent output
        return m_parent->get_input(input_name);
    }
}

utilities::json_t LLMContext::get_inputs() const
{
    utilities::json_t inputs = nlohmann::json::object();

    for (const auto& in_map : m_inputs)
    {
        inputs[in_map.internal_name] = this->get_input(in_map.internal_name);
    }

    return inputs;
}

void LLMContext::set_output(utilities::json_t outputs)
{
    m_outputs = std::move(outputs);
    this->outputs_complete();
}

void LLMContext::set_output(const std::string& output_name, utilities::json_t output)
{
    m_outputs[output_name] = std::move(output);
}

void LLMContext::set_output_names(std::vector<std::string> output_names)
{
    m_output_names = std::move(output_names);
}

void LLMContext::outputs_complete()
{
    // m_outputs_promise.set_value();
}

const utilities::json_t& LLMContext::view_outputs() const
{
    // // Wait for the outputs to be available
    // m_outputs_future.wait();

    return m_outputs;
}

}  // namespace morpheus::llm
