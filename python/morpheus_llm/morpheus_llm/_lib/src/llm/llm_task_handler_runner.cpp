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

#include "morpheus_llm/llm/llm_task_handler_runner.hpp"

#include "morpheus_llm/llm/llm_context.hpp"

#include <glog/logging.h>
#include <mrc/coroutines/task.hpp>  // IWYU pragma: keep

#include <coroutine>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace morpheus::llm {

LLMTaskHandlerRunner::LLMTaskHandlerRunner(input_mappings_t inputs, std::shared_ptr<LLMTaskHandler> handler) :
  m_inputs(std::move(inputs)),
  m_handler(std::move(handler))
{
    // TODO(MDD): Check that the input map is valid

    // Get the inputs of the current node
    auto input_names = m_handler->get_input_names();

    // Replace any placeholders with the real node input name
    for (const auto& inp : m_inputs)
    {
        const auto& node_name  = inp.internal_name;
        const auto& input_name = inp.external_name;

        // Check that the input name and node names are valid
        CHECK_EQ(node_name.find("*"), std::string::npos) << "Invalid node name '" << node_name << "'";
        CHECK_EQ(input_name.find("*"), std::string::npos) << "Invalid input_name '" << input_name << "'";

        CHECK_EQ(node_name.find("-"), std::string::npos) << "Invalid node name '" << node_name << "'";
        CHECK_EQ(input_name.find("-"), std::string::npos) << "Invalid input_name '" << input_name << "'";
    }
}

LLMTaskHandlerRunner::~LLMTaskHandlerRunner() = default;

Task<LLMTaskHandler::return_t> LLMTaskHandlerRunner::try_handle(std::shared_ptr<LLMContext> context)
{
    // Create a new context
    auto child_context = context->push("TaskHandler", m_inputs);

    // Also need error handling here
    co_return co_await m_handler->try_handle(child_context);
}

}  // namespace morpheus::llm
