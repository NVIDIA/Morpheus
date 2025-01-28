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

#include "morpheus_llm/llm/fwd.hpp"  // for ControlMessage, LLMContext, LLMTask, LLMTaskHandler, LLMTaskHandlerRunner
#include "morpheus_llm/llm/input_map.hpp"
#include "morpheus_llm/llm/llm_node.hpp"

#include "morpheus/export.h"
#include "morpheus/types.hpp"

#include <memory>
#include <vector>

namespace morpheus::llm {

/**
 * @brief A subclass of LLMNode that acts as container for components required in processing LLM service
 * requests. For example, a prompt generator and LLM service can each be implemented as an LLMNode and added to
 * the engine. Input mappings are used to match node's input to outputs of parent or sibling nodes. Task handlers
 * can also be added to the engine to perform additional processing on the outputs stored in the LLMContext of the
 * engine.
 */
class MORPHEUS_EXPORT LLMEngine : public LLMNode
{
  public:
    /**
     * @brief Construct a new LLMEngine object.
     *
     */
    LLMEngine();

    /**
     * @brief Destroy the LLMEngine object.
     *
     */
    ~LLMEngine() override;

    /**
     * @brief Add new task handler to this engine.
     *
     * @param inputs input mappings that specifies inputs to new task handler
     * @param task_handler task handler object
     */
    virtual void add_task_handler(user_input_mappings_t inputs, std::shared_ptr<LLMTaskHandler> task_handler);

    /**
     * @brief Execute nodes in this engine and pass outputs to its task handlers. Must pass this a control message with
     * a 'llm_engine' task` containing 'task_type' and 'task_dict' properties required for execution of task(s).
     *
     * @param input_message input control message
     * @return Task<std::vector<std::shared_ptr<ControlMessage>>>
     */
    virtual Task<std::vector<std::shared_ptr<ControlMessage>>> run(std::shared_ptr<ControlMessage> input_message);

  private:
    Task<std::vector<std::shared_ptr<ControlMessage>>> handle_tasks(std::shared_ptr<LLMContext> context);

    std::vector<std::shared_ptr<LLMTaskHandlerRunner>> m_task_handlers;
};

}  // namespace morpheus::llm
