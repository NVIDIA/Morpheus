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

#include "morpheus/messages/memory/inference_memory_fil.hpp"

#include "morpheus/messages/memory/inference_memory.hpp"
#include "morpheus/utilities/cupy_util.hpp"  // for CupyUtil

#include <pybind11/pytypes.h>

#include <memory>
#include <utility>

namespace morpheus {
/****** Component public implementations *******************/
/****** InferenceMemoryFIL****************************************/
InferenceMemoryFIL::InferenceMemoryFIL(TensorIndex count, TensorObject&& input__0, TensorObject&& seq_ids) :
  InferenceMemory(count)
{
    set_tensor("input__0", std::move(input__0));
    set_tensor("seq_ids", std::move(seq_ids));
}

const TensorObject& InferenceMemoryFIL::get_input__0() const
{
    return get_tensor("input__0");
}

void InferenceMemoryFIL::set_input__0(TensorObject&& input__0)
{
    set_tensor("input__0", std::move(input__0));
}

const TensorObject& InferenceMemoryFIL::get_seq_ids() const
{
    return get_tensor("seq_ids");
}

void InferenceMemoryFIL::set_seq_ids(TensorObject&& seq_ids)
{
    set_tensor("seq_ids", std::move(seq_ids));
}

/****** InferenceMemoryFILInterfaceProxy *************************/
std::shared_ptr<InferenceMemoryFIL> InferenceMemoryFILInterfaceProxy::init(TensorIndex count,
                                                                           pybind11::object input__0,
                                                                           pybind11::object seq_ids)
{
    // Convert the cupy arrays to tensors
    return std::make_shared<InferenceMemoryFIL>(
        count, std::move(CupyUtil::cupy_to_tensor(input__0)), std::move(CupyUtil::cupy_to_tensor(seq_ids)));
}

pybind11::object InferenceMemoryFILInterfaceProxy::get_input__0(InferenceMemoryFIL& self)
{
    return get_tensor_property(self, "input__0");
}

void InferenceMemoryFILInterfaceProxy::set_input__0(InferenceMemoryFIL& self, pybind11::object cupy_values)
{
    self.set_input__0(CupyUtil::cupy_to_tensor(cupy_values));
}

pybind11::object InferenceMemoryFILInterfaceProxy::get_seq_ids(InferenceMemoryFIL& self)
{
    return get_tensor_property(self, "seq_ids");
}

void InferenceMemoryFILInterfaceProxy::set_seq_ids(InferenceMemoryFIL& self, pybind11::object cupy_values)
{
    return self.set_seq_ids(CupyUtil::cupy_to_tensor(cupy_values));
}
}  // namespace morpheus
