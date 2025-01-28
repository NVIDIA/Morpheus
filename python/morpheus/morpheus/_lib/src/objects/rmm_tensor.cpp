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

#include "morpheus/objects/rmm_tensor.hpp"

#include "morpheus/objects/dev_mem_info.hpp"  // for DevMemInfo
#include "morpheus/objects/dtype.hpp"
#include "morpheus/objects/tensor_object.hpp"
#include "morpheus/utilities/matx_util.hpp"
#include "morpheus/utilities/tensor_util.hpp"  // for get_elem_count & get_element_stride

#include <cuda_runtime.h>            // for cudaMemcpy, cudaMemcpy2D, cudaMemcpyDeviceToDevice
#include <glog/logging.h>            // for DCHECK_LT, COMPACT_GOOGLE_LOG_FATAL, DCHECK, DCHECK_EQ, LogMessageFatal
#include <mrc/cuda/common.hpp>       // for MRC_CHECK_CUDA
#include <rmm/cuda_stream_view.hpp>  // for cuda_stream_per_thread
#include <rmm/device_buffer.hpp>

#include <algorithm>  // for copy, transform
#include <functional>  // for multiplies, plus, minus
#include <iterator>    // for back_insert_iterator, back_inserter
#include <memory>
#include <numeric>  // for transform_reduce
#include <ostream>  // needed for logging
#include <utility>  // for move, pair
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** RMMTensor****************************************/
RMMTensor::RMMTensor(std::shared_ptr<rmm::device_buffer> device_buffer,
                     TensorSize offset,
                     DType dtype,
                     ShapeType shape,
                     ShapeType stride) :
  m_mem_descriptor(std::make_shared<MemoryDescriptor>(device_buffer->stream(), device_buffer->memory_resource())),
  m_md(std::move(device_buffer)),
  m_offset(offset),
  m_dtype(std::move(dtype)),
  m_shape(std::move(shape)),
  m_stride(std::move(stride))
{
    if (m_stride.empty())
    {
        detail::validate_stride(this->m_shape, this->m_stride);
    }

    DCHECK(m_offset + this->bytes() <= m_md->size())
        << "Inconsistent tensor. Tensor values would extend past the end of the device_buffer";
}

std::shared_ptr<MemoryDescriptor> RMMTensor::get_memory() const
{
    return m_mem_descriptor;
}

void* RMMTensor::data() const
{
    return static_cast<uint8_t*>(m_md->data()) + this->offset_bytes();
}

RankType RMMTensor::rank() const
{
    return m_shape.size();
}

DType RMMTensor::dtype() const
{
    return m_dtype;
}

TensorSize RMMTensor::count() const
{
    return TensorUtils::get_elem_count(m_shape);
}

TensorSize RMMTensor::bytes() const
{
    return count() * m_dtype.item_size();
}

TensorIndex RMMTensor::shape(TensorIndex idx) const
{
    DCHECK_LT(idx, m_shape.size());
    return m_shape.at(idx);
}

TensorIndex RMMTensor::stride(TensorIndex idx) const
{
    DCHECK_LT(idx, m_stride.size());
    return m_stride.at(idx);
}

void RMMTensor::get_shape(ShapeType& s) const
{
    s.resize(rank());
    std::copy(m_shape.begin(), m_shape.end(), s.begin());
}

void RMMTensor::get_stride(ShapeType& s) const
{
    s.resize(rank());
    std::copy(m_stride.begin(), m_stride.end(), s.begin());
}

intptr_t RMMTensor::stream() const
{
    return (intptr_t)m_md->stream().value();
}

bool RMMTensor::is_compact() const
{
    TensorIndex ttl = 1;
    for (int i = rank() - 1; i >= 0; i--)
    {
        if (stride(i) != ttl)
        {
            return false;
        }

        ttl *= shape(i);
    }
    return true;
}

std::shared_ptr<ITensor> RMMTensor::slice(const ShapeType& min_dims, const ShapeType& max_dims) const
{
    // Calc new offset
    auto offset = std::transform_reduce(
        m_stride.begin(), m_stride.end(), min_dims.begin(), m_offset, std::plus<>(), std::multiplies<>());

    // Calc new shape
    ShapeType shape;
    std::transform(max_dims.begin(), max_dims.end(), min_dims.begin(), std::back_inserter(shape), std::minus<>());

    // Stride remains the same

    return std::make_shared<RMMTensor>(m_md, offset, m_dtype, shape, m_stride);
}

std::shared_ptr<ITensor> RMMTensor::reshape(const ShapeType& dims) const
{
    return std::make_shared<RMMTensor>(m_md, 0, m_dtype, dims, m_stride);
}

std::shared_ptr<ITensor> RMMTensor::deep_copy() const
{
    // Deep copy
    std::shared_ptr<rmm::device_buffer> copied_buffer =
        std::make_shared<rmm::device_buffer>(*m_md, m_md->stream(), m_md->memory_resource());

    return std::make_shared<RMMTensor>(copied_buffer, m_offset, m_dtype, m_shape, m_stride);
}

std::shared_ptr<ITensor> RMMTensor::as_type(DType new_dtype) const
{
    // Now do the conversion
    auto new_data_buffer =
        MatxUtil::cast(DevMemInfo{m_md, m_dtype, m_shape, m_stride, this->offset_bytes()}, new_dtype.type_id());

    // Return the new type
    return std::make_shared<RMMTensor>(new_data_buffer, 0, new_dtype, m_shape, m_stride);
}

TensorSize RMMTensor::offset_bytes() const
{
    return m_offset * m_dtype.item_size();
}

std::shared_ptr<ITensor> RMMTensor::copy_rows(const std::vector<RangeType>& selected_rows, TensorIndex num_rows) const
{
    const auto tensor_type = dtype();
    const auto item_size   = tensor_type.item_size();
    const auto num_columns = shape(1);
    const auto stride      = TensorUtils::get_element_stride(m_stride);
    const auto row_stride  = stride[0];

    auto output_buffer =
        std::make_shared<rmm::device_buffer>(num_rows * num_columns * item_size, rmm::cuda_stream_per_thread);

    auto output_offset = static_cast<uint8_t*>(output_buffer->data());

    for (const auto& rows : selected_rows)
    {
        const auto& sliced_input_tensor  = slice({rows.first, 0}, {rows.second, num_columns});
        const TensorIndex num_input_rows = rows.second - rows.first;
        DCHECK_EQ(num_input_rows, sliced_input_tensor->shape(0));

        const auto slice_size = sliced_input_tensor->bytes();

        if (row_stride == 1)
        {
            // column major just use cudaMemcpy
            MRC_CHECK_CUDA(
                cudaMemcpy(output_offset, sliced_input_tensor->data(), slice_size, cudaMemcpyDeviceToDevice));
        }
        else
        {
            MRC_CHECK_CUDA(cudaMemcpy2D(output_offset,
                                        item_size,
                                        sliced_input_tensor->data(),
                                        row_stride * item_size,
                                        item_size,
                                        num_input_rows,
                                        cudaMemcpyDeviceToDevice));
        }

        output_offset += slice_size;
    }

    ShapeType output_shape{num_rows, num_columns};
    return std::make_shared<RMMTensor>(output_buffer, 0, tensor_type, output_shape);
}
}  // namespace morpheus
