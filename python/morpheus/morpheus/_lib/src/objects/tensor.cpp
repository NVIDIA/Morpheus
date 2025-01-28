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

#include "morpheus/objects/tensor.hpp"

#include "morpheus/objects/dtype.hpp"
#include "morpheus/objects/memory_descriptor.hpp"  // for MemoryDescriptor
#include "morpheus/objects/rmm_tensor.hpp"
#include "morpheus/objects/tensor_object.hpp"
#include "morpheus/utilities/tensor_util.hpp"  // for TensorUtils::get_element_stride

#include <cuda_runtime.h>       // for cudaMemcpy, cudaMemcpyDeviceToHost
#include <mrc/cuda/common.hpp>  // for MRC_CHECK_CUDA
#include <rmm/device_buffer.hpp>

#include <algorithm>  // needed by get_element_stride
#include <memory>
#include <string>
#include <utility>  // for move
#include <vector>

namespace morpheus {
Tensor::Tensor(std::shared_ptr<rmm::device_buffer> buffer,
               std::string init_typestr,
               ShapeType init_shape,
               ShapeType init_strides,
               TensorSize init_offset) :
  m_device_buffer(std::move(buffer)),
  typestr(std::move(init_typestr)),
  shape(std::move(init_shape)),
  strides(std::move(init_strides)),
  m_offset(init_offset)
{}

void* Tensor::data() const
{
    return static_cast<uint8_t*>(m_device_buffer->data()) + m_offset;
}

TensorSize Tensor::bytes_count() const
{
    // temp just return without shape, size, offset, etc
    return m_device_buffer->size();
}

std::vector<uint8_t> Tensor::get_host_data() const
{
    std::vector<uint8_t> out_data;

    out_data.resize(this->bytes_count());

    MRC_CHECK_CUDA(cudaMemcpy(&out_data[0], this->data(), this->bytes_count(), cudaMemcpyDeviceToHost));

    return out_data;
}

auto Tensor::get_stream() const
{
    return this->m_device_buffer->stream();
}

TensorObject Tensor::create(
    std::shared_ptr<rmm::device_buffer> buffer, DType dtype, ShapeType shape, ShapeType strides, TensorSize offset)
{
    auto md = std::make_shared<MemoryDescriptor>(buffer->stream(), buffer->memory_resource());

    if (!strides.empty())
    {
        strides = TensorUtils::get_element_stride<TensorIndex>(strides);
    }
    auto tensor = std::make_shared<RMMTensor>(buffer, offset, dtype, shape, strides);

    return {md, tensor};
}
}  // namespace morpheus
