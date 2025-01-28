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
#include "morpheus/objects/dtype.hpp"  // for DType
#include "morpheus/objects/memory_descriptor.hpp"
#include "morpheus/objects/tensor_object.hpp"
#include "morpheus/types.hpp"  // for RankType, ShapeType, TensorIndex, TensorSize

#include <rmm/device_buffer.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace morpheus {

/****** Component public implementations *******************/
/****** RMMTensor****************************************/

/**
 * @addtogroup objects
 * @{
 * @file
 */

/**
 * TODO(Documentation)
 */
class MORPHEUS_EXPORT RMMTensor : public ITensor
{
  public:
    RMMTensor(std::shared_ptr<rmm::device_buffer> device_buffer,
              TensorSize offset,
              DType dtype,
              ShapeType shape,
              ShapeType stride = {});

    ~RMMTensor() override = default;

    /**
     * TODO(Documentation)
     */
    bool is_compact() const final;

    /**
     * TODO(Documentation)
     */
    DType dtype() const override;

    /**
     * TODO(Documentation)
     */
    RankType rank() const final;

    /**
     * TODO(Documentation)
     */
    std::shared_ptr<ITensor> deep_copy() const override;

    /**
     * TODO(Documentation)
     */
    std::shared_ptr<ITensor> reshape(const ShapeType& dims) const override;

    /**
     * TODO(Documentation)
     */
    std::shared_ptr<ITensor> slice(const ShapeType& min_dims, const ShapeType& max_dims) const override;

    /**
     * @brief Creates a depp copy of the specified rows specified as vector<pair<start, stop>> not inclusive
     * of the stop row.
     *
     * @param selected_rows
     * @param num_rows
     * @return std::shared_ptr<ITensor>
     */
    std::shared_ptr<ITensor> copy_rows(const std::vector<RangeType>& selected_rows,
                                       TensorIndex num_rows) const override;

    /**
     * TODO(Documentation)
     */
    std::shared_ptr<MemoryDescriptor> get_memory() const override;

    /**
     * TODO(Documentation)
     */
    TensorSize bytes() const final;

    /**
     * TODO(Documentation)
     */
    TensorSize count() const final;

    /**
     * TODO(Documentation)
     */
    TensorIndex shape(TensorIndex idx) const final;

    /**
     * TODO(Documentation)
     */
    TensorIndex stride(TensorIndex idx) const final;

    /**
     * TODO(Documentation)
     */
    void* data() const override;

    /**
     * TODO(Documentation)
     */
    void get_shape(ShapeType& s) const;

    /**
     * TODO(Documentation)
     */
    void get_stride(ShapeType& s) const;

    intptr_t stream() const override;

    // Tensor reshape(ShapeType shape)
    // {
    //     CHECK(is_compact());
    //     return Tensor(descriptor_shared(), dtype_size(), shape);
    // }

    /**
     * TODO(Documentation)
     */
    std::shared_ptr<ITensor> as_type(DType dtype) const override;

  protected:
  private:
    /**
     * TODO(Documentation)
     */
    TensorSize offset_bytes() const;

    // Memory info
    std::shared_ptr<MemoryDescriptor> m_mem_descriptor;
    std::shared_ptr<rmm::device_buffer> m_md;
    TensorSize m_offset;

    // Type info
    DType m_dtype;

    // Shape info
    ShapeType m_shape;
    ShapeType m_stride;
};
/** @} */  // end of group
}  // namespace morpheus
