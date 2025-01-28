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

#include "morpheus_llm/llm/fwd.hpp"
#include "morpheus_llm/llm/input_map.hpp"
#include "morpheus_llm/llm/llm_task_handler.hpp"

#include "morpheus/export.h"
#include "morpheus/types.hpp"

#include <memory>

namespace morpheus::llm {

/**
 * @brief This class wraps LLMTaskHandler and maintains how its inputs map to node outputs in the LLMEngine.
 */
class MORPHEUS_EXPORT LLMTaskHandlerRunner
{
  public:
    /**
     * @brief Construct a new LLMTaskHandlerRunner object.
     *
     * @param inputs input mappings
     * @param handler task handler object
     */
    LLMTaskHandlerRunner(input_mappings_t inputs, std::shared_ptr<LLMTaskHandler> handler);

    /**
     * @brief Destroy the LLMTaskHandlerRunner object.
     */
    ~LLMTaskHandlerRunner();

    /**
     * @brief Virtual method for implementing task. Called after execution of LLM engine nodes and outputs
     * saved to given context.
     *
     * @param context context holding outputs after execution of engine nodes
     * @return Task<return_t>
     */
    virtual Task<LLMTaskHandler::return_t> try_handle(std::shared_ptr<LLMContext> context);

    /**
     * @brief Get input names for task handler.
     *
     * @return std::vector<std::string>
     */
    const input_mappings_t& input_names() const
    {
        return m_inputs;
    }

  private:
    input_mappings_t m_inputs;
    std::shared_ptr<LLMTaskHandler> m_handler;
};

}  // namespace morpheus::llm
