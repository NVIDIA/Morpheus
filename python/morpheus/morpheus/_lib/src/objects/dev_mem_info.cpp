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

#include "morpheus/objects/dev_mem_info.hpp"

#include "morpheus/utilities/tensor_util.hpp"  // for get_elem_count

#include <glog/logging.h>  // for DCHECK

#include <cstdint>  // for uint8_t
#include <memory>
#include <ostream>
#include <utility>  // for move

namespace morpheus {
// Component public implementations
// ************ DevMemInfo************************* //
DevMemInfo::DevMemInfo(void* data,
                       DType dtype,
                       std::shared_ptr<MemoryDescriptor> md,
                       ShapeType shape,
                       ShapeType stride,
                       TensorSize offset_bytes) :
  m_data(data),
  m_dtype(std::move(dtype)),
  m_md(std::move(md)),
  m_shape(std::move(shape)),
  m_stride(std::move(stride)),
  m_offset_bytes(offset_bytes)
{
    DCHECK(m_md != nullptr);
}

DevMemInfo::DevMemInfo(std::shared_ptr<rmm::device_buffer> buffer,
                       DType dtype,
                       ShapeType shape,
                       ShapeType stride,
                       TensorSize offset_bytes) :
  m_data(buffer->data()),
  m_dtype(std::move(dtype)),
  m_shape(std::move(shape)),
  m_stride(std::move(stride)),
  m_offset_bytes(offset_bytes),
  m_md(std::make_shared<MemoryDescriptor>(buffer->stream(), buffer->memory_resource()))
{
    DCHECK(m_offset_bytes + this->bytes() <= buffer->size())
        << "Inconsistent dimensions, values would extend past the end of the device_buffer";
}

TensorSize DevMemInfo::bytes() const
{
    return count() * m_dtype.item_size();
}

TensorSize DevMemInfo::count() const
{
    return TensorUtils::get_elem_count(m_shape);
}

TensorSize DevMemInfo::offset_bytes() const
{
    return m_offset_bytes;
}

const DType& DevMemInfo::dtype() const
{
    return m_dtype;
}

TypeId DevMemInfo::type_id() const
{
    return m_dtype.type_id();
}

const ShapeType& DevMemInfo::shape() const
{
    return m_shape;
}

TensorIndex DevMemInfo::shape(TensorIndex idx) const
{
    return m_shape.at(idx);
}

const ShapeType& DevMemInfo::stride() const
{
    return m_stride;
}

TensorIndex DevMemInfo::stride(TensorIndex idx) const
{
    return m_stride.at(idx);
}

std::shared_ptr<MemoryDescriptor> DevMemInfo::memory() const
{
    return m_md;
}

std::unique_ptr<rmm::device_buffer> DevMemInfo::make_new_buffer(TensorSize bytes) const
{
    return std::make_unique<rmm::device_buffer>(bytes, m_md->cuda_stream, m_md->memory_resource);
}

void* DevMemInfo::data() const
{
    return static_cast<uint8_t*>(m_data) + m_offset_bytes;
}
}  // namespace morpheus
