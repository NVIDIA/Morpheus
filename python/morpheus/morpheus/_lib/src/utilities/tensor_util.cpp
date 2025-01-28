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

#include "morpheus/utilities/tensor_util.hpp"

#include <glog/logging.h>              // for DCHECK_EQ
#include <mrc/utils/sort_indexes.hpp>  // for sort_indexes
                                       // clang-format off
// prevent from moving this into the third-party section
#include <experimental/iterator>  // for make_ostream_joiner
#include <ostream>      // for operator<<, ostream, stringstream
#include <string>       // for char_traits, string
#include <vector>       // for vector

namespace morpheus {
void TensorUtils::write_shape_to_stream(const ShapeType& shape, std::ostream& os)
{
    os << "(";
    std::copy(shape.begin(), shape.end(), std::experimental::make_ostream_joiner(os, ", "));
    os << ")";
}

std::string TensorUtils::shape_to_string(const ShapeType& shape)
{
    std::stringstream ss;
    write_shape_to_stream(shape, ss);
    return ss.str();
}

void TensorUtils::set_contiguous_stride(const ShapeType& shape, ShapeType& stride)
{
    stride.resize(shape.size());
    TensorIndex ttl = 1;
    auto rank       = shape.size();
    for (int i = rank - 1; i >= 0; i--)
    {
        stride[i] = ttl;
        ttl *= shape.at(i);
    }
}

bool TensorUtils::has_contiguous_stride(const ShapeType& shape, const ShapeType& stride)
{
    DCHECK_EQ(shape.size(), stride.size());
    auto count = get_elem_count(shape);
    return (shape[0] * stride[0] == count);
}

bool TensorUtils::validate_shape_and_stride(const ShapeType& shape,
                                            const ShapeType& stride)
{
    if (shape.size() != stride.size())
    {
        return false;
    }

    auto stride_sorted_idx = mrc::sort_indexes(stride);

    for (int i = 0; i < stride_sorted_idx.size() - 1; ++i)
    {
        if (!(stride[stride_sorted_idx[i]] * shape[stride_sorted_idx[i]] <= stride[stride_sorted_idx[i + 1]]))
        {
            return false;
        }
    }

    return true;
}

}  // namespace morpheus
