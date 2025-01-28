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

#include "py_llm_node.hpp"

#include "morpheus_llm/llm/input_map.hpp"
#include "morpheus_llm/llm/llm_engine.hpp"

#include <mrc/coroutines/task.hpp>  // IWYU pragma: keep
#include <pybind11/pybind11.h>

#include <utility>

namespace morpheus::llm {

template <class BaseT>
std::shared_ptr<LLMNodeRunner> PyLLMNode<BaseT>::add_node(std::string name,
                                                          user_input_mappings_t inputs,
                                                          std::shared_ptr<LLMNodeBase> node,
                                                          bool is_output)
{
    // Try to cast the object to a python object to ensure that we keep it alive
    m_py_nodes[node] = pybind11::cast(node);

    // Call the base class implementation
    return LLMNode::add_node(std::move(name), std::move(inputs), std::move(node), is_output);
}

// explicit instantiations
template class PyLLMNode<>;
template class PyLLMNode<LLMEngine>;

}  // namespace morpheus::llm
