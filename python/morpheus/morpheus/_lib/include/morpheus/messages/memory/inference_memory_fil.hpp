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
#include "morpheus/messages/memory/inference_memory.hpp"
#include "morpheus/objects/tensor_object.hpp"
#include "morpheus/types.hpp"  // for TensorIndex

#include <pybind11/pytypes.h>  // for object

#include <memory>

namespace morpheus {
/****** Component public implementations *******************/
/****** InferenceMemoryFIL****************************************/

/**
 * @addtogroup messages
 * @{
 * @file
 */
/**
 * This is a container class for data that needs to be submitted to the inference server for FIL category
 * usecases.
 */
class MORPHEUS_EXPORT InferenceMemoryFIL : public InferenceMemory
{
  public:
    /**
     * @brief Construct a new Inference Memory FIL object
     *
     * @param count : Message count in inference memory object
     * @param input__0 : Inference input
     * @param seq_ids : Ids used to index from an inference input to a message. Necessary since there can be more
     * inference inputs than messages (i.e., if some messages get broken into multiple inference requests)
     */
    InferenceMemoryFIL(TensorIndex count, TensorObject&& input__0, TensorObject&& seq_ids);

    /**
     * @brief Returns the 'input__0' tensor, throws a `std::runtime_error` if it does not exist
     *
     * @throw std::runtime_error
     * @return const TensorObject&
     */
    const TensorObject& get_input__0() const;

    /**
     * @brief Returns the 'seq_ids' tensor, throws a `std::runtime_error` if it does not exist
     *
     * @throw std::runtime_error
     * @return const TensorObject&
     */
    const TensorObject& get_seq_ids() const;

    /**
     * @brief Sets a tensor named 'input__0'
     *
     * @param input__0
     * @throws std::length_error If the number of rows in `input__0` does not match `count`.
     */
    void set_input__0(TensorObject&& input__0);

    /**
     * @brief Sets a tensor named 'seq_ids'
     *
     * @param seq_ids
     * @throws std::length_error If the number of rows in `seq_ids` does not match `count`.
     */
    void set_seq_ids(TensorObject&& seq_ids);
};

/****** InferenceMemoryFILInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings
 */
struct MORPHEUS_EXPORT InferenceMemoryFILInterfaceProxy : public InferenceMemoryInterfaceProxy
{
    /**
     * @brief Create and initialize an InferenceMemoryFIL object, and return a shared pointer to the result
     *
     * @param count : Message count in inference memory object
     * @param input__0 : Inference input
     * @param seq_ids : Ids used to index from an inference input to a message. Necessary since there can be more
     * inference inputs than messages (i.e., if some messages get broken into multiple inference requests)
     * @return std::shared_ptr<InferenceMemoryFIL>
     */
    static std::shared_ptr<InferenceMemoryFIL> init(TensorIndex count,
                                                    pybind11::object input__0,
                                                    pybind11::object seq_ids);

    /**
     * @brief Returns the 'input__0' as cupy array
     *
     * @param self
     * @return pybind11::object
     */
    static pybind11::object get_input__0(InferenceMemoryFIL& self);

    /**
     * @brief Sets a tensor named 'input__0'
     *
     * @param self
     * @param cupy_values
     */
    static void set_input__0(InferenceMemoryFIL& self, pybind11::object cupy_values);

    /**
     * @brief Returns the 'seq_ids' as a cupy array
     *
     * @param self
     * @return pybind11::object
     */
    static pybind11::object get_seq_ids(InferenceMemoryFIL& self);

    /**
     * @brief Sets a tensor named 'seq_ids'
     *
     * @param self
     * @param cupy_values
     */
    static void set_seq_ids(InferenceMemoryFIL& self, pybind11::object cupy_values);
};
/** @} */  // end of group
}  // namespace morpheus
