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

#pragma once

#include "morpheus/export.h"
#include "morpheus/messages/memory/response_memory.hpp"
#include "morpheus/objects/tensor_object.hpp"
#include "morpheus/types.hpp"  // for TensorMap

#include <pybind11/pytypes.h>

#include <memory>

namespace morpheus {
/****** Component public implementations *******************/
/****** ResponseMemoryProbs*********************************/

/**
 * @addtogroup messages
 * @{
 * @file
 */

/**
 * @brief Output memory block containing the inference response probabilities.
 *
 */
class MORPHEUS_EXPORT ResponseMemoryProbs : public ResponseMemory
{
  public:
    /**
     * @brief Construct a new Response Memory Probs object
     *
     * @param count
     * @param probs
     */
    ResponseMemoryProbs(TensorIndex count, TensorObject&& probs);
    /**
     * @brief Construct a new Response Memory Probs object
     *
     * @param count
     * @param tensors
     */
    ResponseMemoryProbs(TensorIndex count, TensorMap&& tensors);

    /**
     * @brief Returns the tensor named 'probs'. alias for `get_tensor("probs")`
     *
     * @return const TensorObject&
     * @throws std::runtime_error If no tensor named "probs" exists
     */
    const TensorObject& get_probs() const;

    /**
     * @brief Update the tensor named 'probs'
     *
     * @param probs
     * @throws std::length_error If the number of rows in `probs` does not match `count`.
     */
    void set_probs(TensorObject&& probs);
};

/****** ResponseMemoryProbsInterfaceProxy*******************/

/**
 * @brief Interface proxy, used to insulate python bindings
 */
struct MORPHEUS_EXPORT ResponseMemoryProbsInterfaceProxy : public ResponseMemoryInterfaceProxy
{
    /**
     * @brief Create and initialize a ResponseMemoryProbs object, and return a shared pointer to the result
     *
     * @param count
     * @param probs
     * @return std::shared_ptr<ResponseMemoryProbs>
     */
    static std::shared_ptr<ResponseMemoryProbs> init(TensorIndex count, pybind11::object probs);

    /**
     * @brief Get the response memory probs object ()
     *
     * @param self
     * @return pybind11::object
     * @throws pybind11::key_error When no tensor named "probs" exists.
     */
    static pybind11::object get_probs(ResponseMemoryProbs& self);

    /**
     * @brief Set the response memory probs object (alias for `set_tensor("probs", cupy_values)`)
     *
     * @param self
     * @param cupy_values
     */
    static void set_probs(ResponseMemoryProbs& self, pybind11::object cupy_values);
};

/** @} */  // end of group
}  // namespace morpheus
