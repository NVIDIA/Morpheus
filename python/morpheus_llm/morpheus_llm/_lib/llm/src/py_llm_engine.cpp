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

#include "py_llm_engine.hpp"

#include "morpheus_llm/llm/llm_task_handler.hpp"  // for LLMTaskHandler
#include "py_llm_task_handler.hpp"

#include <pybind11/pybind11.h>

#include <utility>

namespace morpheus::llm {

PyLLMEngine::PyLLMEngine() : PyLLMNode<LLMEngine>() {}

PyLLMEngine::~PyLLMEngine() = default;

void PyLLMEngine::add_task_handler(user_input_mappings_t inputs, std::shared_ptr<LLMTaskHandler> task_handler)
{
    // Try to cast the object to a python object to ensure that we keep it alive
    auto py_task_handler = std::dynamic_pointer_cast<PyLLMTaskHandler>(task_handler);

    if (py_task_handler)
    {
        // Store the python object to keep it alive
        m_py_task_handler[task_handler] = pybind11::cast(task_handler);
    }

    // Call the base class implementation
    LLMEngine::add_task_handler(std::move(inputs), std::move(task_handler));
}

}  // namespace morpheus::llm
