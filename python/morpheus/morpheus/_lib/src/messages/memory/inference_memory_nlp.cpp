/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/messages/memory/inference_memory_nlp.hpp"

#include "morpheus/messages/memory/inference_memory.hpp"
#include "morpheus/utilities/cupy_util.hpp"  // for CupyUtil

#include <pybind11/pytypes.h>

#include <utility>  // for move, pair

namespace morpheus {
/****** Component public implementations *******************/
/****** InferenceMemoryNLP ****************************************/
InferenceMemoryNLP::InferenceMemoryNLP(TensorIndex count,
                                       TensorObject&& input_ids,
                                       TensorObject&& input_mask,
                                       TensorObject&& seq_ids) :
  InferenceMemory(count)
{
    set_tensor("input_ids", std::move(input_ids));
    set_tensor("input_mask", std::move(input_mask));
    set_tensor("seq_ids", std::move(seq_ids));
}

const TensorObject& InferenceMemoryNLP::get_input_ids() const
{
    return get_tensor("input_ids");
}

void InferenceMemoryNLP::set_input_ids(TensorObject&& input_ids)
{
    set_tensor("input_ids", std::move(input_ids));
}

const TensorObject& InferenceMemoryNLP::get_input_mask() const
{
    return get_tensor("input_mask");
}

void InferenceMemoryNLP::set_input_mask(TensorObject&& input_mask)
{
    set_tensor("input_mask", std::move(input_mask));
}

const TensorObject& InferenceMemoryNLP::get_seq_ids() const
{
    return get_tensor("seq_ids");
}

void InferenceMemoryNLP::set_seq_ids(TensorObject&& seq_ids)
{
    set_tensor("seq_ids", std::move(seq_ids));
}

/****** InferenceMemoryNLPInterfaceProxy *************************/
std::shared_ptr<InferenceMemoryNLP> InferenceMemoryNLPInterfaceProxy::init(TensorIndex count,
                                                                           pybind11::object input_ids,
                                                                           pybind11::object input_mask,
                                                                           pybind11::object seq_ids)
{
    // Convert the cupy arrays to tensors
    return std::make_shared<InferenceMemoryNLP>(count,
                                                std::move(CupyUtil::cupy_to_tensor(input_ids)),
                                                std::move(CupyUtil::cupy_to_tensor(input_mask)),
                                                std::move(CupyUtil::cupy_to_tensor(seq_ids)));
}

pybind11::object InferenceMemoryNLPInterfaceProxy::get_input_ids(InferenceMemoryNLP& self)
{
    return get_tensor_property(self, "input_ids");
}

void InferenceMemoryNLPInterfaceProxy::set_input_ids(InferenceMemoryNLP& self, pybind11::object cupy_values)
{
    self.set_input_ids(CupyUtil::cupy_to_tensor(cupy_values));
}

pybind11::object InferenceMemoryNLPInterfaceProxy::get_input_mask(InferenceMemoryNLP& self)
{
    return get_tensor_property(self, "input_mask");
}

void InferenceMemoryNLPInterfaceProxy::set_input_mask(InferenceMemoryNLP& self, pybind11::object cupy_values)
{
    return self.set_input_mask(CupyUtil::cupy_to_tensor(cupy_values));
}

pybind11::object InferenceMemoryNLPInterfaceProxy::get_seq_ids(InferenceMemoryNLP& self)
{
    return get_tensor_property(self, "seq_ids");
}

void InferenceMemoryNLPInterfaceProxy::set_seq_ids(InferenceMemoryNLP& self, pybind11::object cupy_values)
{
    return self.set_seq_ids(CupyUtil::cupy_to_tensor(cupy_values));
}
}  // namespace morpheus
