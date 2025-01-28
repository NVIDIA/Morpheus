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

#include "morpheus/messages/memory/response_memory_probs.hpp"

#include "morpheus/utilities/cupy_util.hpp"

#include <glog/logging.h>
#include <pybind11/pytypes.h>

#include <memory>
#include <ostream>
#include <utility>

namespace morpheus {
/****** Component public implementations *******************/
/****** ResponseMemoryProbs****************************************/
ResponseMemoryProbs::ResponseMemoryProbs(TensorIndex count, TensorObject&& probs) : ResponseMemory(count)
{
    set_tensor("probs", std::move(probs));
}

ResponseMemoryProbs::ResponseMemoryProbs(TensorIndex count, TensorMap&& tensors) :
  ResponseMemory(count, std::move(tensors))
{
    CHECK(has_tensor("probs")) << "Tensor: 'probs' not found in memory";
}

const TensorObject& ResponseMemoryProbs::get_probs() const
{
    return get_tensor("probs");
}

void ResponseMemoryProbs::set_probs(TensorObject&& probs)
{
    set_tensor("probs", std::move(probs));
}

/****** ResponseMemoryProbsInterfaceProxy *************************/
std::shared_ptr<ResponseMemoryProbs> ResponseMemoryProbsInterfaceProxy::init(TensorIndex count, pybind11::object probs)
{
    // Conver the cupy arrays to tensors
    return std::make_shared<ResponseMemoryProbs>(count, std::move(CupyUtil::cupy_to_tensor(probs)));
}

pybind11::object ResponseMemoryProbsInterfaceProxy::get_probs(ResponseMemoryProbs& self)
{
    return get_tensor_property(self, "probs");
}

void ResponseMemoryProbsInterfaceProxy::set_probs(ResponseMemoryProbs& self, pybind11::object cupy_values)
{
    self.set_probs(CupyUtil::cupy_to_tensor(cupy_values));
}
}  // namespace morpheus
