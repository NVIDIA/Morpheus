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
#include "morpheus/objects/dtype.hpp"
#include "morpheus/objects/tensor_object.hpp"
#include "morpheus/types.hpp"  // for ShapeType, TensorIndex, TensorSize

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cstdint>  // for uint8_t
#include <memory>
#include <string>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** Tensor****************************************/

/**
 * @addtogroup objects
 * @{
 * @file
 */

/**
 * @brief Morpheus Tensor object, using RMM device buffers for the underlying storage. Typically created using the
 * `Tensor::create` factory method and accessed through a `TensorObject` handle.
 *
 */

class MORPHEUS_EXPORT Tensor
{
  public:
    Tensor(std::shared_ptr<rmm::device_buffer> buffer,
           std::string init_typestr,
           ShapeType init_shape,
           ShapeType init_strides,
           TensorSize init_offset = 0);

    ShapeType shape;
    ShapeType strides;
    std::string typestr;

    /**
     * TODO(Documentation)
     */
    void* data() const;

    /**
     * TODO(Documentation)
     */
    TensorSize bytes_count() const;

    /**
     * TODO(Documentation)
     */
    std::vector<uint8_t> get_host_data() const;

    /**
     * TODO(Documentation)
     */
    auto get_stream() const;

    /**
     * TODO(Documentation)
     */
    static TensorObject create(std::shared_ptr<rmm::device_buffer> buffer,
                               DType dtype,
                               ShapeType shape,
                               ShapeType strides,
                               TensorSize offset = 0);

  private:
    TensorSize m_offset;
    std::shared_ptr<rmm::device_buffer> m_device_buffer;
};
/** @} */  // end of group
}  // namespace morpheus
