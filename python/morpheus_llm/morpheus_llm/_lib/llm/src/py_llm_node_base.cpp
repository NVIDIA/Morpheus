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

#include "py_llm_node_base.hpp"

#include "morpheus_llm/llm/llm_context.hpp"  // IWYU pragma: keep
#include "morpheus_llm/llm/llm_engine.hpp"
#include "morpheus_llm/llm/llm_node.hpp"
#include "morpheus_llm/llm/llm_node_base.hpp"

#include <glog/logging.h>
#include <mrc/coroutines/task.hpp>  // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // IWYU pragma: keep
#include <pymrc/coro.hpp>  // IWYU pragma: keep

#include <coroutine>

namespace morpheus::llm {

template <class BaseT>
PyLLMNodeBase<BaseT>::~PyLLMNodeBase() = default;

template <class BaseT>
std::vector<std::string> PyLLMNodeBase<BaseT>::get_input_names() const
{
    MRC_PYBIND11_OVERRIDE_PURE_TEMPLATE(std::vector<std::string>, LLMNodeBase, BaseT, get_input_names);
}

template <class BaseT>
Task<std::shared_ptr<LLMContext>> PyLLMNodeBase<BaseT>::execute(std::shared_ptr<LLMContext> context)
{
    MRC_PYBIND11_OVERRIDE_CORO_PURE_TEMPLATE(std::shared_ptr<LLMContext>, LLMNodeBase, BaseT, execute, context);
}

// explicit instantiations
template class PyLLMNodeBase<>;  // LLMNodeBase
template class PyLLMNodeBase<LLMNode>;
template class PyLLMNodeBase<LLMEngine>;

}  // namespace morpheus::llm
