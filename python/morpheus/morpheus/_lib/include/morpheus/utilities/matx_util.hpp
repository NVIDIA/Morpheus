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

#include "morpheus/objects/dev_mem_info.hpp"
#include "morpheus/objects/dtype.hpp"
#include "morpheus/objects/rmm_tensor.hpp"
#include "morpheus/objects/tensor_object.hpp"
#include "morpheus/types.hpp"  // for ShapeType, TensorIndex

#include <memory>
#include <vector>

namespace morpheus {

/**
 * @addtogroup utilities
 * @{
 * @file
 */

struct MatxUtil
{
    /**
     * @brief Convert one device_buffer type to another
     *
     * @param input
     * @param output_type
     * @return std::shared_ptr<rmm::device_buffer>
     */
    static std::shared_ptr<rmm::device_buffer> cast(const DevMemInfo& input, TypeId output_type);

    /**
     * @brief Builds a Nx3 segment ID matrix
     *
     * @param row_count
     * @param fea_len
     * @param output_type
     * @param start_idx
     * @return std::shared_ptr<rmm::device_buffer>
     */
    static std::shared_ptr<rmm::device_buffer> create_seq_ids(TensorIndex row_count,
                                                              TensorIndex fea_len,
                                                              TypeId output_type,
                                                              std::shared_ptr<MemoryDescriptor> md,
                                                              TensorIndex start_idx = 0);

    /**
     * @brief Adds a constant offset to a seg_ids tensor
     *
     * @param input
     * @param offset
     */
    static void offset_seq_ids(const DevMemInfo& input, TensorIndex offset);

    /**
     * @brief Calculate logits on device_buffer
     *
     * @param input
     * @return std::shared_ptr<rmm::device_buffer>
     */
    static std::shared_ptr<rmm::device_buffer> logits(const DevMemInfo& input);

    /**
     * @brief Perform transpose
     *
     * @param input
     * @return std::shared_ptr<rmm::device_buffer>
     */
    static std::shared_ptr<rmm::device_buffer> transpose(const DevMemInfo& input);

    /**
     * @brief Returns an array of boolean where x[i,j] >= thresh_val, when by_row is true an Nx1 array will be returned
     * with a true if any value in the row is above the threshold
     *
     * @param input
     * @param thresh_val
     * @param by_row
     * @return std::shared_ptr<rmm::device_buffer>
     */
    static std::shared_ptr<rmm::device_buffer> threshold(const DevMemInfo& input, double thresh_val, bool by_row);

    /**
     * @brief Returns a buffer with `output_shape` containing the max value from values in `input` mapped according to
     * `seq_ids`.
     * Ex given a hypothetical input of:
     *
     *     input =   [5, 2, 8, 9, 8, 2, 1]
     *     seq_ids = [0, 0, 0, 1, 2, 3, 3]
     *
     * Will return:
     *               [8, 9, 8, 2]
     *
     * @param input
     * @param seq_ids
     * @param seq_id_offset
     * @param output_shape
     * @return std::shared_ptr<rmm::device_buffer>
     */
    static std::shared_ptr<rmm::device_buffer> reduce_max(const DevMemInfo& input,
                                                          const ShapeType& seq_ids,
                                                          TensorIndex seq_id_offset,
                                                          const ShapeType& output_shape);
};
/** @} */  // end of group
}  // namespace morpheus
