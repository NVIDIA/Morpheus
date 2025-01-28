/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "morpheus/stages/add_scores_stage_base.hpp"

#include "morpheus/messages/memory/tensor_memory.hpp"  // for TensorMemory
#include "morpheus/messages/meta.hpp"                  // for MessageMeta
#include "morpheus/objects/dtype.hpp"                  // for DType
#include "morpheus/objects/tensor.hpp"                 // for Tensor
#include "morpheus/objects/tensor_object.hpp"          // for TensorObject
#include "morpheus/types.hpp"                          // for TensorIndex
#include "morpheus/utilities/matx_util.hpp"            // for MatxUtil
#include "morpheus/utilities/string_util.hpp"          // for StringUtil
#include "morpheus/utilities/tensor_util.hpp"          // for TensorUtils

#include <glog/logging.h>  // for CHECK, COMPACT_GOOGLE_LOG_FATAL, LogMessageFatal
#include <rxcpp/rx.hpp>    // for observable_member, trace_activity, decay_t, map, opera...

#include <cstddef>   // for size_t
#include <iterator>  // for reverse_iterator
#include <memory>    // for shared_ptr, allocator, __shared_ptr_access
#include <ostream>   // for basic_ostream, operator<<, basic_ostream::operator<<
#include <utility>   // for move, pair
#include <vector>    // for vector

namespace morpheus {

// Component public implementations
// *********************** AddScoresStageBase *********************** //
AddScoresStageBase::AddScoresStageBase(std::map<std::size_t, std::string> idx2label, std::optional<float> threshold) :
  base_t(),
  m_idx2label(std::move(idx2label)),
  m_threshold(threshold),
  m_min_col_count(m_idx2label.rbegin()->first)  // Ordered map's largest key will be the last entry
{
    this->pipe(rxcpp::operators::map([this](sink_type_t msg) {
        return this->on_data(std::move(msg));
    }));
}

AddScoresStageBase::source_type_t AddScoresStageBase::on_data(sink_type_t msg)
{
    // The default of probs_tensor_name is "probs"
    auto probs        = msg->tensors()->get_tensor("probs");
    const auto& shape = probs.get_shape();

    // Depending on the input the stride is given in bytes or elements, convert to
    // elements
    auto stride = TensorUtils::get_element_stride(probs.get_stride());

    CHECK(shape.size() == 2 && shape[1] > m_min_col_count)
        << "Model output did not contain enough columns to fufill the requested "
           "labels. Label "
           "indexes: "
        << StringUtil::map_to_str(m_idx2label.begin(), m_idx2label.end()) << ", Model output columns: " << shape[1];

    const auto num_rows    = shape[0];
    const auto num_columns = shape[1];

    TensorObject output_tensor;

    if (m_threshold.has_value())
    {
        auto thresh_bool_buffer = MatxUtil::threshold(
            {probs.data(), probs.dtype(), probs.get_memory(), probs.get_shape(), probs.get_stride()},
            *m_threshold,
            false);

        output_tensor.swap(Tensor::create(thresh_bool_buffer, DType::create<bool>(), shape, stride));
    }
    else
    {
        output_tensor.swap(std::move(probs));
    }

    std::vector<std::string> columns;
    std::vector<TensorObject> tensors;

    std::size_t i = 0;
    for (const auto& [column_num, column_name] : m_idx2label)
    {
        columns.push_back(column_name);
        tensors.emplace_back(output_tensor.slice({0, static_cast<TensorIndex>(column_num)},
                                                 {num_rows, static_cast<TensorIndex>(column_num + 1)}));

        ++i;
    }

    msg->payload()->set_data(columns, tensors);
    return msg;
}

}  // namespace morpheus
