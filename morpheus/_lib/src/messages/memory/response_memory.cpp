/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/messages/memory/response_memory.hpp"

#include "morpheus/utilities/cupy_util.hpp"

#include <string>
#include <utility>  // for move

namespace morpheus {
/****** Component public implementations *******************/
/****** ResponseMemory****************************************/
ResponseMemory::ResponseMemory(size_t count) : TensorMemory(count) {}
ResponseMemory::ResponseMemory(size_t count, CupyUtil::tensor_map_t&& tensors) : TensorMemory(count, std::move(tensors))
{}

bool ResponseMemory::has_output(const std::string& name) const
{
    return this->has_tensor(name);
}

/****** ResponseMemoryInterfaceProxy *************************/
std::shared_ptr<ResponseMemory> ResponseMemoryInterfaceProxy::init(std::size_t count, CupyUtil::py_tensor_map_t tensors)
{
    return std::make_shared<ResponseMemory>(count, std::move(CupyUtil::cupy_to_tensors(tensors)));
}

}  // namespace morpheus
