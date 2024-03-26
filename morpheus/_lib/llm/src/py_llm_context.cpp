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

#include "py_llm_context.hpp"

#include "morpheus/llm/llm_context.hpp"
#include "morpheus/utilities/string_util.hpp"  // for MORPHEUS_CONCAT_STR

#include <pybind11/pybind11.h>
#include <pymrc/utilities/json_values.hpp>
#include <pymrc/utils.hpp>  // for cast_from_json

#include <memory>

namespace morpheus::llm {
namespace py = pybind11;

py::object PyLLMContext::all_outputs() const
{
    return m_outputs.to_python();
}

std::shared_ptr<LLMContext> PyLLMContext::push(std::string name, input_mappings_t inputs)
{
    return std::make_shared<PyLLMContext>(this->shared_from_this(), std::move(name), std::move(inputs));
}

void PyLLMContext::pop()
{
    auto py_parent = std::dynamic_pointer_cast<PyLLMContext>(m_parent);
    if (py_parent)
    {
        pybind11::gil_scoped_acquire gil;
        auto outputs = m_outputs.to_python();

        // Copy the outputs from the child context to the parent
        if (m_output_names.empty())
        {
            // Use them all by default
            py_parent->set_output(m_name, std::move(outputs));
        }
        else if (m_output_names.size() == 1)
        {
            // Treat only a single output as the output
            py_parent->set_output(m_name, outputs.attr("pop")(m_output_names[0].c_str()));
        }
        else
        {
            // Build a new json object with only the specified keys
            py::dict new_outputs;

            for (const auto& output_name : m_output_names)
            {
                new_outputs[output_name.c_str()] = outputs.attr("pop")(output_name.c_str());
            }

            py_parent->set_output(m_name, std::move(new_outputs));
        }

        m_outputs = std::move(mrc::pymrc::JSONValues(std::move(outputs)));
    }
    else
    {
        LLMContext::pop();
    }
}

py::object PyLLMContext::get_py_input() const
{
    if (m_inputs.size() > 1)
    {
        throw std::runtime_error(
            "PyLLMContext::get_input() called on a context with multiple inputs. Use get_input(input_name) instead.");
    }

    return this->get_py_input(m_inputs[0].internal_name);
}

py::object PyLLMContext::get_py_input(const std::string& node_name) const
{
    pybind11::gil_scoped_acquire gil;
    if (node_name[0] == '/')
    {
        try
        {
            return m_outputs.get_python(node_name);
        } catch (py::error_already_set& err)
        {
            throw std::runtime_error(MORPHEUS_CONCAT_STR("Input '" << node_name << "' not found in the output map"));
        }
    }

    auto found       = find_input(node_name);
    auto& input_name = found->external_name;

    // Get the value from a parent output
    auto py_parent = std::dynamic_pointer_cast<PyLLMContext>(m_parent);
    if (py_parent)
    {
        return py_parent->get_py_input(input_name);
    }

    auto json_input = m_parent->get_input(input_name);
    return mrc::pymrc::cast_from_json(json_input).cast<py::dict>();
}

py::object PyLLMContext::get_py_inputs() const
{
    py::dict inputs;
    for (const auto& in_map : m_inputs)
    {
        inputs[in_map.internal_name.c_str()] = this->get_py_input(in_map.internal_name);
    }

    return inputs;
}

py::object PyLLMContext::view_outputs() const
{
    return this->all_outputs();
}

void PyLLMContext::set_output(py::object outputs)
{
    mrc::pymrc::JSONValues json_values(outputs);
    LLMContext::set_output(std::move(json_values));
}

void PyLLMContext::set_output(const std::string& output_name, py::object output)
{
    mrc::pymrc::JSONValues json_value(output);
    LLMContext::set_output(output_name, std::move(json_value));
}

}  // namespace morpheus::llm
