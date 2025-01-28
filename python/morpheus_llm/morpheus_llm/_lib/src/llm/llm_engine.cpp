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

#include "morpheus_llm/llm/llm_engine.hpp"

#include "morpheus_llm/llm/llm_context.hpp"  // for LLMContext
#include "morpheus_llm/llm/llm_task.hpp"
#include "morpheus_llm/llm/llm_task_handler.hpp"         // for LLMTaskHandler
#include "morpheus_llm/llm/llm_task_handler_runner.hpp"  // for LLMTaskHandlerRunner
#include "morpheus_llm/llm/utils.hpp"

#include "morpheus/messages/control.hpp"      // for ControlMessage
#include "morpheus/utilities/json_types.hpp"  // for PythonByteContainer

#include <mrc/coroutines/task.hpp>  // IWYU pragma: keep
#include <nlohmann/json.hpp>

#include <algorithm>  // for max (implied usage)
#include <coroutine>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace morpheus::llm {

LLMEngine::LLMEngine() = default;

LLMEngine::~LLMEngine() = default;

void LLMEngine::add_task_handler(user_input_mappings_t inputs, std::shared_ptr<LLMTaskHandler> task_handler)
{
    auto input_names = task_handler->get_input_names();

    auto final_inputs = process_input_names(inputs, input_names);

    m_task_handlers.push_back(std::make_shared<LLMTaskHandlerRunner>(std::move(final_inputs), task_handler));
}

Task<std::vector<std::shared_ptr<ControlMessage>>> LLMEngine::run(std::shared_ptr<ControlMessage> input_message)
{
    if (!input_message)
    {
        throw std::runtime_error("LLMEngine::run() called with a null message");
    }

    if (!input_message->has_task("llm_engine"))
    {
        throw std::runtime_error("LLMEngine::run() called with a message that does not have the 'llm_engine' task");
    }

    std::vector<std::shared_ptr<ControlMessage>> output_messages;

    while (input_message->has_task("llm_engine"))
    {
        auto current_task = input_message->remove_task("llm_engine");

        // Temp create an instance of LLMTask for type safety
        LLMTask tmp_task(current_task["task_type"].get<std::string>(), current_task.at("task_dict"));

        // Set the name, task, control_message and inputs on the context
        auto context = std::make_shared<LLMContext>(tmp_task, input_message);

        // Call the base node
        co_await this->execute(context);

        // Pass the outputs into the task generators
        auto tasks = co_await this->handle_tasks(context);

        output_messages.insert(output_messages.end(), tasks.begin(), tasks.end());
    }

    co_return output_messages;
}

Task<std::vector<std::shared_ptr<ControlMessage>>> LLMEngine::handle_tasks(std::shared_ptr<LLMContext> context)
{
    // Wait for the base node outputs (This will yield if not already available)
    // auto outputs = context->get_outputs();

    for (auto& task_handler : m_task_handlers)
    {
        auto new_tasks = co_await task_handler->try_handle(context);

        if (new_tasks.has_value())
        {
            co_return new_tasks.value();
        }
    }

    throw std::runtime_error("No task handler was able to handle the input message and responses generated");
}

}  // namespace morpheus::llm
