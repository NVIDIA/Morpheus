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

#include "morpheus/messages/memory/inference_memory.hpp"

#include "morpheus/objects/tensor_object.hpp"  // IWYU pragma: keep
#include "morpheus/utilities/cupy_util.hpp"    // for CupyUtil::cupy_to_tensors, CupyUtil::py_tensor_map_t

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <string>
#include <utility>  // for move

namespace morpheus {
/****** Component public implementations *******************/
/****** InferenceMemory****************************************/
InferenceMemory::InferenceMemory(TensorIndex count) : TensorMemory(count) {}
InferenceMemory::InferenceMemory(TensorIndex count, TensorMap&& tensors) : TensorMemory(count, std::move(tensors)) {}

bool InferenceMemory::has_input(const std::string& name) const
{
    return this->has_tensor(name);
}

/****** InferenceMemoryInterfaceProxy *************************/
std::shared_ptr<InferenceMemory> InferenceMemoryInterfaceProxy::init(TensorIndex count, pybind11::object& tensors)
{
    if (tensors.is_none())
    {
        return std::make_shared<InferenceMemory>(count);
    }
    else
    {
        return std::make_shared<InferenceMemory>(
            count, std::move(CupyUtil::cupy_to_tensors(tensors.cast<CupyUtil::py_tensor_map_t>())));
    }
}

}  // namespace morpheus
