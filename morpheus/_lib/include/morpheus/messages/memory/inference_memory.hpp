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

#pragma once

#include "morpheus/messages/memory/tensor_memory.hpp"
#include "morpheus/types.hpp"  // for TensorMap

#include <pybind11/pytypes.h>  // for object

#include <cstddef>  // for size_t
#include <memory>   // for shared_ptr
#include <string>

namespace morpheus {
/**
 * @addtogroup messages
 * @{
 * @file
 */

#pragma GCC visibility push(default)
/**
 * @brief This is a base container class for data that will be used for inference stages. This class is designed to
    hold generic data as a `TensorObject`s
 *
 */
class InferenceMemory : public TensorMemory
{
  public:
    /**
     * @brief Construct a new Inference Memory object
     *
     * @param count
     */
    InferenceMemory(size_t count);
    /**
     * @brief Construct a new Inference Memory object
     *
     * @param count
     * @param tensors
     */
    InferenceMemory(size_t count, TensorMap&& tensors);

    /**
     * @brief Checks if a tensor named `name` exists in `tensors`. Alias for `has_tensor`.
     *
     * @param name
     * @return true
     * @return false
     */
    bool has_input(const std::string& name) const;
};

/****** InferenceMemoryInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct InferenceMemoryInterfaceProxy : public TensorMemoryInterfaceProxy
{
    /**
     * @brief Create and initialize a InferenceMemory object, and return a shared pointer to the result. Each array in
     * `tensors` should be of length `count`.
     *
     * @param count : Lenght of each array in `tensors`
     * @param tensors : Map of string on to cupy arrays
     * @return std::shared_ptr<InferenceMemory>
     */
    static std::shared_ptr<InferenceMemory> init(std::size_t count, pybind11::object& tensors);
};
#pragma GCC visibility pop

/** @} */  // end of group
}  // namespace morpheus
