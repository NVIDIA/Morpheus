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

#include "morpheus_llm/llm/fwd.hpp"  // for ControlMessage
#include "morpheus_llm/llm/input_map.hpp"
#include "morpheus_llm/llm/llm_task.hpp"

#include "morpheus/export.h"
#include "morpheus/utilities/json_types.hpp"

#include <mrc/types.hpp>

#include <memory>
#include <string>
#include <vector>

namespace morpheus::llm {

struct LLMContextState
{
    LLMTask task;
    std::shared_ptr<ControlMessage> message;
};

/**
 * @brief Holds and manages information related to LLM tasks and input mappings required for LLMNode execution.
 * Outputs of node executions are also saved here for use by nodes and task handlers in LLMEngine.
 */
class MORPHEUS_EXPORT LLMContext : public std::enable_shared_from_this<LLMContext>
{
  public:
    /**
     * @brief Construct a new LLMContext object.
     *
     */
    LLMContext();

    /**
     * @brief Construct a new LLMContext object.
     *
     * @param task task for new context
     * @param message control message for new context
     */
    LLMContext(LLMTask task, std::shared_ptr<ControlMessage> message);

    /**
     * @brief Construct a new LLMContext object.
     *
     * @param parent parent context
     * @param name new context name
     * @param inputs input mappings for new context
     */
    LLMContext(std::shared_ptr<LLMContext> parent, std::string name, input_mappings_t inputs);

    /**
     * @brief Destroy the LLMContext object.
     */
    ~LLMContext();

    /**
     * @brief Get parent context.
     *
     * @return std::shared_ptr<LLMContext>
     */
    std::shared_ptr<LLMContext> parent() const;

    /**
     * @brief Get name of context.
     *
     * @return const std::string&
     */
    const std::string& name() const;

    /**
     * @brief Get map of internal mappings for this context.
     *
     * @return const input_mappings_t&
     */
    const input_mappings_t& input_map() const;

    /**
     * @brief Get task for this context.
     *
     * @return const LLMTask&
     */
    const LLMTask& task() const;

    /**
     * @brief Get control message for this context.
     *
     * @return std::shared_ptr<ControlMessage>&
     */
    std::shared_ptr<ControlMessage>& message() const;

    /**
     * @brief Get all outputs for this context.
     *
     * @return const utilities::json_t&
     */
    const utilities::json_t& all_outputs() const;

    /**
     * @brief Get full name of context containing parents up to root.
     *
     * @return std::string
     */
    std::string full_name() const;

    /**
     * @brief Create new context from this context using provided name and input mappings.
     *
     * @param name name of new context
     * @param inputs input mappings for new context
     * @return std::shared_ptr<LLMContext>
     */
    std::shared_ptr<LLMContext> push(std::string name, input_mappings_t inputs);

    /**
     * @brief Moves output map from this context to parent context. Outputs to move can be selected using
     * set_output_names, otherwise all outputs are noved by default.
     *
     */
    void pop();

    /**
     * @brief Get the input value from parent context corresponding to first internal input of this context.
     *
     * @return utilities::json_t
     */
    const utilities::json_t& get_input() const;

    /**
     * @brief Get the parent output value corresponding to given internal input name.
     *
     * @param node_name internal input name
     * @return utilities::json_t
     */
    const utilities::json_t& get_input(const std::string& node_name) const;

    /**
     * @brief Get parent output values corresponding to all internal input names.
     *
     * @return utilities::json_t
     */
    utilities::json_t get_inputs() const;

    /**
     * @brief Set output mappings for this context.
     *
     * @param outputs output mappings
     */
    void set_output(utilities::json_t outputs);

    /**
     * @brief Set an output value for this context.
     *
     * @param output_name output name
     * @param output output value
     */
    void set_output(const std::string& output_name, utilities::json_t output);

    /**
     * @brief Set the output names to propagate from this context when using pop.
     *
     * @param output_names output names to propagate
     */
    void set_output_names(std::vector<std::string> output_names);

    void outputs_complete();

    /**
     * @brief Get all outputs for this context.
     *
     * @return const utilities::json_t&
     */
    const utilities::json_t& view_outputs() const;

  private:
    input_mappings_t::const_iterator find_input(const std::string& node_name, bool throw_if_not_found = true) const;

    std::shared_ptr<LLMContext> m_parent{nullptr};
    std::string m_name;
    input_mappings_t m_inputs;
    std::vector<std::string> m_output_names;  // Names of keys to be used as the output. Empty means use all keys

    std::shared_ptr<LLMContextState> m_state;

    utilities::json_t m_outputs;

    mrc::Promise<void> m_outputs_promise;
    mrc::SharedFuture<void> m_outputs_future;
};
}  // namespace morpheus::llm
