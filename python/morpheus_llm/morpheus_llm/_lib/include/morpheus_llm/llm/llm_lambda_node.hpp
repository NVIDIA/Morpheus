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

#include "morpheus_llm/llm/llm_context.hpp"
#include "morpheus_llm/llm/llm_node_base.hpp"

#include "morpheus/export.h"
#include "morpheus/types.hpp"
#include "morpheus/utilities/type_traits.hpp"

#include <boost/type_traits/function_traits.hpp>
#include <mrc/coroutines/task.hpp>
#include <mrc/type_traits.hpp>
#include <nlohmann/json_fwd.hpp>

#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace morpheus::llm {

/**
 * @brief Class template for a LLMNode created from a function that returns a Task.
 *
 * @tparam ReturnT return type
 * @tparam ArgsT arguments type
 */
template <typename ReturnT, typename... ArgsT>
class LLMLambdaNode : public LLMNodeBase
{
  public:
    using function_t = std::function<Task<ReturnT>(ArgsT...)>;

    LLMLambdaNode(std::vector<std::string> input_names, function_t function) :
      m_input_names(std::move(input_names)),
      m_function(std::move(function))
    {}

    std::vector<std::string> get_input_names() const override
    {
        return m_input_names;
    }

    Task<std::shared_ptr<LLMContext>> execute(std::shared_ptr<LLMContext> context) override
    {
        using args_tuple_t = std::tuple<ArgsT...>;

        if constexpr (std::tuple_size<args_tuple_t>::value == 0)
        {
            auto outputs = co_await this->m_function();

            nlohmann::json outputs_json = std::move(outputs);

            // Set the outputs
            context->set_output(std::move(outputs_json));

            co_return context;
        }
        else if constexpr (std::tuple_size<args_tuple_t>::value == 1)
        {
            const auto& arg = context->get_input();

            auto output = co_await this->m_function(arg.get<std::tuple_element_t<0, args_tuple_t>>());

            nlohmann::json outputs_json = std::move(output);

            // Set the outputs
            context->set_output(std::move(outputs_json));

            co_return context;
        }
        else
        {
            auto args = context->get_inputs();

            auto outputs = co_await this->m_function(args);

            nlohmann::json outputs_json = std::move(outputs);

            // Set the outputs
            context->set_output(std::move(outputs_json));

            co_return context;
        }
    }

  protected:
    std::vector<std::string> m_input_names;
    function_t m_function;
};

/**
 * @brief Template function that creates a LLMLambdaNode from a function.
 *
 * @tparam ReturnT return type
 * @tparam ArgsT function args
 * @param fn function that returns a Task
 * @return auto
 */
template <typename ReturnT, typename... ArgsT>
auto make_lambda_node(std::function<ReturnT(ArgsT...)>&& fn)
{
    using function_t = std::function<ReturnT(ArgsT...)>;

    static_assert(utilities::is_specialization<typename function_t::result_type, mrc::coroutines::Task>::value,
                  "Return type must be a Task");

    using return_t = typename utilities::extract_value_type<typename function_t::result_type>::type;

    auto make_args = []<std::size_t... Is>(std::index_sequence<Is...>) {
        return std::vector<std::string>{std::string{"arg"} + std::to_string(Is)...};
    };

    return std::make_shared<LLMLambdaNode<return_t, ArgsT...>>(make_args(std::index_sequence_for<ArgsT...>{}),
                                                               std::move(fn));
}

/**
 * @brief Template function that creates a LLMLambdaNode from a lambda or function pointer.
 *
 * @tparam FuncT function type
 * @param fn function that returns a Task
 * @return auto
 */
template <typename FuncT>
auto make_lambda_node(FuncT&& fn)
{
    // Convert the incoming object to a function in case its a lambda or C* function pointer
    return make_lambda_node(std::function{std::forward<FuncT>(fn)});
}

}  // namespace morpheus::llm
