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

#include "./test_utils/common.hpp"  // IWYU pragma: associated

#include "morpheus/objects/dev_mem_info.hpp"
#include "morpheus/objects/dtype.hpp"  // for DType
#include "morpheus/objects/memory_descriptor.hpp"
#include "morpheus/types.hpp"  // for ShapeType, TensorIndex

#include <cuda/memory_resource>
#include <gtest/gtest.h>  // for AssertionResult, SuiteApiResolver, TestInfo, EXPECT_TRUE, Message, TEST_F, Test, TestFactoryImpl, TestPartResult
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/arena_memory_resource.hpp>       // for arena_memory_resource
#include <rmm/mr/device/cuda_memory_resource.hpp>        // for cuda_memory_resource
#include <rmm/mr/device/fixed_size_memory_resource.hpp>  // for fixed_size_memory_resource
#include <rmm/mr/device/owning_wrapper.hpp>              // for make_owning_wrapper
#include <sys/types.h>

#include <cstddef>  // for size_t
#include <memory>   // shared_ptr
// IWYU pragma: no_include "thrust/iterator/iterator_facade.h"
// IWYU pragma: no_include <unordered_map>

using namespace morpheus;

namespace {
const TensorIndex ROWS = 20;
const TensorIndex COLS = 5;
const auto DTYPE       = DType::create<float>();

const auto BYTE_SIZE = ROWS * COLS * DTYPE.item_size();

template <template <typename> class ResourceT>
auto make_mem_resource()
{
    return rmm::mr::make_owning_wrapper<ResourceT>(std::make_shared<rmm::mr::cuda_memory_resource>());
}
}  // namespace

TEST_CLASS(DevMemInfo);

TEST_F(TestDevMemInfo, RmmBufferConstructor)
{
    // Explicitly passing the non-default values to make sure we don't have a test that accidentally passes
    auto mem_resource = make_mem_resource<rmm::mr::fixed_size_memory_resource>();
    auto buffer       = std::make_shared<rmm::device_buffer>(BYTE_SIZE, rmm::cuda_stream_legacy, mem_resource.get());

    // Set the offset to the second row in the buffer
    DevMemInfo dm{buffer, DTYPE, {ROWS - 1, COLS}, {1, ROWS}, DTYPE.item_size()};

    EXPECT_EQ(dm.bytes(), (ROWS - 1) * COLS * DTYPE.item_size());
    EXPECT_EQ(dm.count(), (ROWS - 1) * COLS);
    EXPECT_EQ(dm.offset_bytes(), DTYPE.item_size());

    EXPECT_EQ(dm.dtype(), DTYPE);
    EXPECT_EQ(dm.type_id(), DTYPE.type_id());

    EXPECT_EQ(dm.shape().size(), 2);
    EXPECT_EQ(dm.shape()[0], ROWS - 1);
    EXPECT_EQ(dm.shape(0), ROWS - 1);
    EXPECT_EQ(dm.shape()[1], COLS);
    EXPECT_EQ(dm.shape(1), COLS);

    EXPECT_EQ(dm.stride().size(), 2);
    EXPECT_EQ(dm.stride()[0], 1);
    EXPECT_EQ(dm.stride(0), 1);
    EXPECT_EQ(dm.stride()[1], ROWS);
    EXPECT_EQ(dm.stride(1), ROWS);

    EXPECT_EQ(dm.data(), static_cast<u_int8_t*>(buffer->data()) + DTYPE.item_size());

    EXPECT_EQ(dm.memory()->cuda_stream, rmm::cuda_stream_legacy);
    EXPECT_EQ(dm.memory()->memory_resource,
              static_cast<cuda::mr::async_resource_ref<cuda::mr::device_accessible>>(mem_resource.get()));
}

TEST_F(TestDevMemInfo, VoidPtrConstructor)
{
    // Explicitly passing the non-default values to make sure we don't have a test that accidentally passes
    auto mem_resource = make_mem_resource<rmm::mr::arena_memory_resource>();
    auto buffer       = std::make_shared<rmm::device_buffer>(BYTE_SIZE, rmm::cuda_stream_legacy, mem_resource.get());

    auto md = std::make_shared<MemoryDescriptor>(rmm::cuda_stream_legacy, mem_resource.get());

    // Set the offset to the second row in the buffer
    DevMemInfo dm{buffer->data(), DTYPE, md, {ROWS - 1, COLS}, {1, ROWS}, DTYPE.item_size()};

    EXPECT_EQ(dm.bytes(), (ROWS - 1) * COLS * DTYPE.item_size());
    EXPECT_EQ(dm.count(), (ROWS - 1) * COLS);
    EXPECT_EQ(dm.offset_bytes(), DTYPE.item_size());

    EXPECT_EQ(dm.dtype(), DTYPE);
    EXPECT_EQ(dm.type_id(), DTYPE.type_id());

    EXPECT_EQ(dm.shape().size(), 2);
    EXPECT_EQ(dm.shape()[0], ROWS - 1);
    EXPECT_EQ(dm.shape(0), ROWS - 1);
    EXPECT_EQ(dm.shape()[1], COLS);
    EXPECT_EQ(dm.shape(1), COLS);

    EXPECT_EQ(dm.stride().size(), 2);
    EXPECT_EQ(dm.stride()[0], 1);
    EXPECT_EQ(dm.stride(0), 1);
    EXPECT_EQ(dm.stride()[1], ROWS);
    EXPECT_EQ(dm.stride(1), ROWS);

    EXPECT_EQ(dm.data(), static_cast<u_int8_t*>(buffer->data()) + DTYPE.item_size());

    EXPECT_EQ(dm.memory()->cuda_stream, rmm::cuda_stream_legacy);
    EXPECT_EQ(dm.memory()->memory_resource,
              static_cast<cuda::mr::async_resource_ref<cuda::mr::device_accessible>>(mem_resource.get()));
}

TEST_F(TestDevMemInfo, MakeNewBuffer)
{
    // Explicitly passing the non-default values to make sure we don't have a test that accidentally passes
    auto mem_resource = make_mem_resource<rmm::mr::fixed_size_memory_resource>();
    auto buffer       = std::make_shared<rmm::device_buffer>(BYTE_SIZE, rmm::cuda_stream_legacy, mem_resource.get());

    DevMemInfo dm{buffer, DTYPE, {ROWS, COLS}, {1, ROWS}};

    const std::size_t buff_size = 20;
    auto new_buff               = dm.make_new_buffer(buff_size);
    EXPECT_EQ(new_buff->size(), buff_size);
    EXPECT_EQ(new_buff->stream(), rmm::cuda_stream_legacy);
    EXPECT_EQ(new_buff->memory_resource(),
              static_cast<cuda::mr::async_resource_ref<cuda::mr::device_accessible>>(mem_resource.get()));
}
