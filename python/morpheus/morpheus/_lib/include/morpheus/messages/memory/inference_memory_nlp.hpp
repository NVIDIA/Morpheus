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

#include <pybind11/pytypes.h>

#include <memory>

namespace morpheus {
/****** Component public implementations *******************/
/****** InferenceMemoryNLP**********************************/
/**
 * @addtogroup messages
 * @{
 * @file
 */

/**
 * @brief This is a container class for data that needs to be submitted to the inference server for NLP category
    usecases.
 *
 */
class MORPHEUS_EXPORT InferenceMemoryNLP : public InferenceMemory
{
  public:
    /**
     * @brief Construct a new Inference Memory NLP object
     *
     * @param count : Number of messages
     * @param input_ids : The token-ids for each string padded with 0s to max_length
     * @param input_mask : The mask for token-ids result where corresponding positions identify valid token-id values
     * @param seq_ids : Ids used to index from an inference input to a message. Necessary since there can be more
     inference inputs than messages (i.e., if some messages get broken into multiple inference requests)
     */
    InferenceMemoryNLP(TensorIndex count, TensorObject&& input_ids, TensorObject&& input_mask, TensorObject&& seq_ids);

    /**
     * @brief Get the input ids object
     *
     * @return const TensorObject&
     * @throws std::runtime_error If no tensor named "input_ids" exists
     */
    const TensorObject& get_input_ids() const;

    /**
     * @brief Get the input mask object
     *
     * @return const TensorObject&
     * @throws std::runtime_error If no tensor named "input_mask" exists
     */
    const TensorObject& get_input_mask() const;

    /**
     * @brief Get the seq ids object
     *
     * @return const TensorObject&
     * @throws std::runtime_error If no tensor named "seq_ids" exists
     */
    const TensorObject& get_seq_ids() const;

    /**
     * @brief Set the input ids object
     *
     * @param input_ids
     * @throws std::length_error If the number of rows in `input_ids` does not match `count`.
     */
    void set_input_ids(TensorObject&& input_ids);

    /**
     * @brief Set the input mask object
     *
     * @param input_mask
     * @throws std::length_error If the number of rows in `input_mask` does not match `count`.
     */
    void set_input_mask(TensorObject&& input_mask);

    /**
     * @brief Set the seq ids object
     *
     * @param seq_ids
     * @throws std::length_error If the number of rows in `seq_ids` does not match `count`.
     */
    void set_seq_ids(TensorObject&& seq_ids);
};

/****** InferenceMemoryNLPInterfaceProxy********************/

/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MORPHEUS_EXPORT InferenceMemoryNLPInterfaceProxy : public InferenceMemoryInterfaceProxy
{
    /**
     * @brief Create and initialize an InferenceMemoryNLP object, and return a shared pointer to the result
     *
     * @param count : Message count in inference memory object
     * @param input_ids : The token-ids for each string padded with 0s to max_length
     * @param input_mask : The mask for token-ids result where corresponding positions identify valid token-id values
     * @param seq_ids : Ids used to index from an inference input to a message. Necessary since there can be more
     inference inputs than messages (i.e., if some messages get broken into multiple inference requests)
     * @return std::shared_ptr<InferenceMemoryNLP>
     */
    static std::shared_ptr<InferenceMemoryNLP> init(TensorIndex count,
                                                    pybind11::object input_ids,
                                                    pybind11::object input_mask,
                                                    pybind11::object seq_ids);

    /**
     * @brief : Returns token-ids for each string padded with 0s to max_length as python object
     *
     * @param self
     * @return pybind11::object
     * @throws pybind11::attribute_error
     */
    static pybind11::object get_input_ids(InferenceMemoryNLP& self);

    /**
     * @brief Set the input ids object
     *
     * @param self
     * @param cupy_values
     */
    static void set_input_ids(InferenceMemoryNLP& self, pybind11::object cupy_values);

    /**
     * @brief Get the input mask object
     *
     * @param self
     * @return pybind11::object
     * @throws pybind11::attribute_error
     */
    static pybind11::object get_input_mask(InferenceMemoryNLP& self);

    /**
     * @brief Set the input mask object
     *
     * @param self
     * @param cupy_values
     */
    static void set_input_mask(InferenceMemoryNLP& self, pybind11::object cupy_values);

    /**
     * @brief Get the seq ids object
     *
     * @param self
     * @return pybind11::object
     * @throws pybind11::attribute_error
     */
    static pybind11::object get_seq_ids(InferenceMemoryNLP& self);

    /**
     * @brief Set the seq ids object
     *
     * @param self
     * @param cupy_values
     */
    static void set_seq_ids(InferenceMemoryNLP& self, pybind11::object cupy_values);
};
/** @} */  // end of group
}  // namespace morpheus
