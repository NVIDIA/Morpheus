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

#include "morpheus_llm/llm/fwd.hpp"  // for LLMTaskHandler, ControlMessage, LLMContext, LLMEngine, LLMTask
#include "morpheus_llm/llm/input_map.hpp"
#include "morpheus_llm/llm/llm_engine.hpp"
#include "py_llm_node.hpp"

#include <pybind11/pytypes.h>

#include <map>
#include <memory>

namespace morpheus::llm {

class PyLLMEngine : public PyLLMNode<LLMEngine>
{
  public:
    PyLLMEngine();

    ~PyLLMEngine() override;

    void add_task_handler(user_input_mappings_t inputs, std::shared_ptr<LLMTaskHandler> task_handler) override;

  private:
    // Keep the python objects alive by saving references in this object
    std::map<std::shared_ptr<LLMTaskHandler>, pybind11::object> m_py_task_handler;
};

}  // namespace morpheus::llm
