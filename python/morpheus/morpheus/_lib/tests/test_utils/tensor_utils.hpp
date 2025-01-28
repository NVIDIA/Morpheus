/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime.h>  // for cudaMemcpy, cudaMemcpyKind
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>  // for data_type
#include <cudf/utilities/type_dispatcher.hpp>
#include <glog/logging.h>  // IWYU pragma: keep
#include <gtest/gtest.h>   // IWYU pragma: keep
#include <mrc/cuda/common.hpp>
#include <rmm/device_buffer.hpp>

#include <ostream>  // for char_traits, operator<<, basic_ostream
#include <vector>

namespace morpheus::test {

template <typename T>
auto convert_to_host(const rmm::device_buffer& buffer)
{
    std::vector<T> host_buffer(buffer.size() / sizeof(T));

    MRC_CHECK_CUDA(cudaMemcpy(host_buffer.data(), buffer.data(), buffer.size(), cudaMemcpyDeviceToHost));

    return host_buffer;
}

template <typename T>
auto convert_to_host(const cudf::column_view& buffer)
{
    CHECK(buffer.type().id() == cudf::type_to_id<T>()) << "Column has different type than requested";

    std::vector<T> host_buffer(buffer.size());

    MRC_CHECK_CUDA(cudaMemcpy(host_buffer.data(), buffer.data<T>(), buffer.size() * sizeof(T), cudaMemcpyDeviceToHost));

    return host_buffer;
}

template <typename T>
void assert_eq_device_to_host(const rmm::device_buffer& device, const std::vector<T>& host)
{
    std::vector<T> device_on_host = convert_to_host<T>(device);

    ASSERT_EQ(device_on_host, host);
}

template <typename T>
void assert_eq_device_to_host(const cudf::column_view& device, const std::vector<T>& host)
{
    std::vector<T> device_on_host = convert_to_host<T>(device);

    ASSERT_EQ(device_on_host, host);
}

template <typename T>
void assert_eq_device_to_device(const cudf::column_view& device1, const cudf::column_view& device2)
{
    ASSERT_EQ(device1.size(), device2.size()) << "Columns have different sizes";
    ASSERT_EQ(device1.type(), device2.type()) << "Columns have different types";

    std::vector<T> device1_on_host = convert_to_host<T>(device1);
    std::vector<T> device2_on_host = convert_to_host<T>(device2);

    ASSERT_EQ(device1_on_host, device2_on_host);
}

}  // namespace morpheus::test
