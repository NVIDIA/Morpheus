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

#include "morpheus/stages/add_classification.hpp"

#include "morpheus/objects/dev_mem_info.hpp"  // for DevMemInfo
#include "morpheus/objects/dtype.hpp"         // for DType
#include "morpheus/objects/tensor.hpp"
#include "morpheus/objects/tensor_object.hpp"  // for TensorObject
#include "morpheus/types.hpp"                  // for TensorIndex
#include "morpheus/utilities/matx_util.hpp"
#include "morpheus/utilities/string_util.hpp"
#include "morpheus/utilities/tensor_util.hpp"  // for TensorUtils::get_element_stride

#include <cuda_runtime.h>  // for cudaMemcpy, cudaMemcpyDeviceToDevice
#include <glog/logging.h>
#include <mrc/cuda/common.hpp>       // for MRC_CHECK_CUDA
#include <rmm/cuda_stream_view.hpp>  // for cuda_stream_per_thread
#include <rmm/device_buffer.hpp>     // for device_buffer

#include <cstddef>
#include <exception>
#include <functional>  // for divides, bind, placeholders
#include <iterator>
#include <memory>
#include <ostream>  // needed for logging
#include <utility>  // for move
// IWYU thinks we need __alloc_traits<>::value_type for vector assignments
// IWYU pragma: no_include <ext/alloc_traits.h>

namespace morpheus {
// Component public implementations
// ************ AddClassificationStage **************************** //
AddClassificationsStage::AddClassificationsStage(std::map<std::size_t, std::string> idx2label, float threshold) :
  PythonNode(base_t::op_factory_from_sub_fn(build_operator())),
  m_idx2label(std::move(idx2label)),
  m_threshold(threshold),
  m_min_col_count(m_idx2label.rbegin()->first)  // Ordered map's largest key will be the last entry
{}

AddClassificationsStage::subscribe_fn_t AddClassificationsStage::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output](sink_type_t x) {
                const auto& probs = x->get_probs_tensor();
                const auto& shape = probs.get_shape();

                // Depending on the input the stride is given in bytes or elements, convert to elements
                auto stride = TensorUtils::get_element_stride(probs.get_stride());

                CHECK(shape.size() == 2 && shape[1] > m_min_col_count)
                    << "Model output did not contain enough columns to fufill the requested labels. Label "
                       "indexes: "
                    << StringUtil::map_to_str(m_idx2label.begin(), m_idx2label.end())
                    << ", Model output columns: " << shape[1];

                const auto num_rows    = shape[0];
                const auto num_columns = shape[1];

                // A bit ugly, but we cant get access to the rmm::device_buffer here. So make a copy
                auto tmp_buffer = std::make_shared<rmm::device_buffer>(probs.bytes(), rmm::cuda_stream_per_thread);

                MRC_CHECK_CUDA(
                    cudaMemcpy(tmp_buffer->data(), probs.data(), tmp_buffer->size(), cudaMemcpyDeviceToDevice));

                // Now call the threshold function
                auto thresh_bool_buffer =
                    MatxUtil::threshold(DevMemInfo{tmp_buffer, probs.dtype(), shape, stride}, m_threshold, false);

                auto tensor_obj = Tensor::create(thresh_bool_buffer, DType::create<bool>(), shape, stride);

                std::vector<std::string> columns(m_idx2label.size());
                std::vector<TensorObject> tensors(m_idx2label.size());

                std::size_t i = 0;
                for (const auto& [column_num, column_name] : m_idx2label)
                {
                    columns[i] = column_name;
                    tensors[i] = tensor_obj.slice({0, static_cast<TensorIndex>(column_num)},
                                                  {num_rows, static_cast<TensorIndex>(column_num + 1)});

                    ++i;
                }

                x->set_meta(columns, tensors);

                output.on_next(x);
            },
            [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
            [&]() { output.on_completed(); }));
    };
}

// ************ AddClassificationStageInterfaceProxy ************* //
std::shared_ptr<mrc::segment::Object<AddClassificationsStage>> AddClassificationStageInterfaceProxy::init(
    mrc::segment::Builder& builder,
    const std::string& name,
    std::map<std::size_t, std::string> idx2label,
    float threshold)
{
    auto stage = builder.construct_object<AddClassificationsStage>(name, idx2label, threshold);

    return stage;
}
}  // namespace morpheus
