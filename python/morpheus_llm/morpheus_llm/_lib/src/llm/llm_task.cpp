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

#include "morpheus_llm/llm/llm_task.hpp"

#include <utility>

namespace morpheus::llm {

LLMTask::LLMTask() = default;

LLMTask::LLMTask(std::string task_type, nlohmann::json task_dict) :
  task_type(std::move(task_type)),
  task_dict(std::move(task_dict))
{}

LLMTask::~LLMTask() = default;

size_t LLMTask::size() const
{
    return this->task_dict.size();
}

const nlohmann::json& LLMTask::get(const std::string& key) const
{
    return this->task_dict.at(key);
}

void LLMTask::set(const std::string& key, nlohmann::json&& value)
{
    this->task_dict[key] = std::move(value);
}

}  // namespace morpheus::llm
