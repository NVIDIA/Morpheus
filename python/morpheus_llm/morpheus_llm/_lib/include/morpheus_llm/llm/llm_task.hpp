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

#include "morpheus/export.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <string>

namespace morpheus::llm {

/**
 * @brief Holds information about LLM task. This information is extracted by LLMEngine from
 * input control message and saved to the context for use by task handler(s).
 */
struct MORPHEUS_EXPORT LLMTask
{
    /**
     * @brief Construct a new LLMTask object.
     */
    LLMTask();

    /**
     * @brief Construct a new LLMTask object.
     *
     * @param task_type
     * @param task_dict
     */
    LLMTask(std::string task_type, nlohmann::json task_dict);

    /**
     * @brief Destroy the LLMTask object
     *
     */
    ~LLMTask();

    /**
     * @brief task type
     *
     */
    std::string task_type;

    /**
     * @brief Get size of 'task_dict'.
     *
     * @return size_t
     */
    size_t size() const;

    /**
     * @brief Get value in 'task_dict` corresponding to given key.
     *
     * @param key key of value to get from 'task_dict'
     * @return const nlohmann::json&
     */
    const nlohmann::json& get(const std::string& key) const;

    /**
     * @brief Set or update item in 'task_dict'.
     *
     * @param key key of value in 'task_dict' to set
     * @param value new value
     */
    void set(const std::string& key, nlohmann::json&& value);

    /**
     * @brief dictionary of task-related items used by task handler(s)
     */
    nlohmann::json task_dict;
};

}  // namespace morpheus::llm
