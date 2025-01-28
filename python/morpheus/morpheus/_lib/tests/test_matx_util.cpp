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

#include "morpheus/io/deserializers.hpp"
#include "morpheus/objects/dev_mem_info.hpp"
#include "morpheus/objects/dtype.hpp"
#include "morpheus/types.hpp"  // for ShapeType, TensorIndex
#include "morpheus/utilities/matx_util.hpp"

#include <cuda_runtime.h>               // for cudaMemcpy, cudaMemcpyDeviceToHost, cudaMemcpyHostToDevice
#include <cudf/column/column.hpp>       // for column
#include <cudf/column/column_view.hpp>  // for column_view
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>  // for data_type, size_type
#include <gtest/gtest.h>
#include <mrc/cuda/common.hpp>       // for MRC_CHECK_CUDA
#include <rmm/cuda_stream_view.hpp>  // for cuda_stream_per_thread
#include <rmm/device_buffer.hpp>

#include <cstdint>  // for int64_t, uint8_t
#include <memory>   // for shared_ptr, make_shared, unique_ptr
#include <vector>

using namespace morpheus;

TEST_CLASS(MatxUtil);

TEST_F(TestMatxUtil, ReduceMax1d)
{
    // Test mimics example from the method's docstring
    std::vector<float> input{5, 2, 8, 9, 8, 2, 1};
    ShapeType seq_ids{0, 0, 0, 1, 2, 3, 3};
    std::vector<float> expected_output{8, 9, 8, 2};

    DType dtype(TypeId::FLOAT32);

    auto input_buffer =
        std::make_shared<rmm::device_buffer>(input.size() * dtype.item_size(), rmm::cuda_stream_per_thread);

    MRC_CHECK_CUDA(cudaMemcpy(input_buffer->data(), input.data(), input_buffer->size(), cudaMemcpyHostToDevice));

    DevMemInfo dm{input_buffer, dtype, {static_cast<TensorIndex>(input.size()), 1}, {1, 0}};
    ShapeType output_shape{static_cast<TensorIndex>(expected_output.size()), 1};
    auto output_buffer = MatxUtil::reduce_max(dm, seq_ids, 0, output_shape);

    std::vector<float> output(expected_output.size());
    MRC_CHECK_CUDA(cudaMemcpy(output.data(), output_buffer->data(), output_buffer->size(), cudaMemcpyDeviceToHost));

    EXPECT_EQ(output, expected_output);
}

TEST_F(TestMatxUtil, ReduceMax2dRowMajor)
{
    // clang-format off
    // disabling clang-format to illustrate row-major layout
    std::vector<double> input{
        0.1, 0.7, 0.7, 0.7,
        1.0, 0.9, 0.5, 0.9,
        1.0, 0.6, 0.7, 0.9,
        1.0, 0.2, 0.2, 0.9,
        0.5, 0.8, 0.6, 0.0,
        0.3, 0.4, 0.1, 0.4,
        0.9, 0.3, 1.0, 0.6,
        0.5, 0.5, 0.6, 0.8,
        0.0, 0.3, 0.5, 0.6,
        0.6, 1.0, 0.8, 0.7,
        0.8, 0.8, 1.0, 0.6,
        0.1, 0.9, 0.1, 0.3};

    // reducing 12 rows down to 5
    ShapeType seq_ids{0, 0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4};

    std::vector<double> expected_output{
        1.0, 0.9, 0.7, 0.9,
        1.0, 0.6, 0.7, 0.9,
        1.0, 0.8, 1.0, 0.9,
        0.8, 1.0, 1.0, 0.8,
        0.1, 0.9, 0.1, 0.3};
    // clang-format on

    // Copy data from table into one big buffer
    TensorIndex num_cols      = 4;
    TensorIndex num_rows      = 12;
    TensorIndex expected_rows = expected_output.size() / num_cols;

    EXPECT_EQ(num_cols * num_rows, input.size());
    EXPECT_EQ(expected_rows, 5);
    DType dtype(TypeId::FLOAT64);
    EXPECT_EQ(dtype.item_size(), sizeof(double));

    std::size_t buff_size = input.size() * dtype.item_size();
    auto input_buffer     = std::make_shared<rmm::device_buffer>(buff_size, rmm::cuda_stream_per_thread);

    MRC_CHECK_CUDA(cudaMemcpy(input_buffer->data(), input.data(), input_buffer->size(), cudaMemcpyHostToDevice));

    DevMemInfo dm{input_buffer, dtype, {num_rows, num_cols}, {num_cols, 1}};
    ShapeType output_shape{expected_rows, num_cols};
    auto output_buffer = MatxUtil::reduce_max(dm, seq_ids, 0, output_shape);

    EXPECT_EQ(output_buffer->size(), expected_rows * num_cols * dtype.item_size());

    std::vector<double> output(expected_rows * num_cols);
    MRC_CHECK_CUDA(cudaMemcpy(output.data(), output_buffer->data(), output_buffer->size(), cudaMemcpyDeviceToHost));

    EXPECT_EQ(output.size(), expected_output.size());
    for (std::size_t i = 0; i < output.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(output[i], expected_output[i]);
    }
}

TEST_F(TestMatxUtil, ReduceMax2dColMajor)
{
    auto morpheus_root = test::get_morpheus_root();
    auto input_file    = morpheus_root / "tests/tests_data/filter_probs.csv";

    auto table_m  = morpheus::load_table_from_file(input_file);
    auto num_rows = table_m.tbl->num_rows();
    auto num_cols = table_m.tbl->num_columns();

    EXPECT_EQ(num_rows, 20);
    EXPECT_EQ(num_cols, 4);

    // Copy data from table into one big buffer
    auto dtype            = DType::from_cudf(table_m.tbl->get_column(0).type().id());
    std::size_t buff_size = num_cols * num_rows * dtype.item_size();

    EXPECT_EQ(dtype.item_size(), sizeof(double));
    auto input_buffer = std::make_shared<rmm::device_buffer>(buff_size, rmm::cuda_stream_per_thread);

    std::size_t offset{0};
    for (cudf::size_type i = 0; i < num_cols; ++i)
    {
        auto cv = table_m.tbl->get_column(i).view();
        MRC_CHECK_CUDA(cudaMemcpy(static_cast<uint8_t*>(input_buffer->data()) + offset,
                                  cv.data<uint8_t>(),
                                  num_rows * dtype.item_size(),
                                  cudaMemcpyDeviceToDevice));

        offset += num_rows * dtype.item_size();
    }

    EXPECT_EQ(offset, buff_size);

    // reducing 20 rows down to 12
    ShapeType seq_ids{0, 0, 1, 2, 2, 2, 2, 3, 4, 5, 6, 6, 7, 7, 7, 8, 9, 9, 10, 11};
    // disabling formatting so I can enter the literal values by column
    // clang-format off
    std::vector<double> expected_output{0.1, 1.0, 1.0, 1.0, 0.5, 0.3, 0.9, 0.5, 0.0, 0.6, 0.8, 0.1,
                                        0.7, 0.9, 0.6, 0.2, 0.8, 0.4, 0.3, 0.5, 0.3, 1.0, 0.8, 0.9,
                                        0.7, 0.5, 0.7, 0.2, 0.6, 0.1, 1.0, 0.6, 0.5, 0.8, 1.0, 0.1,
                                        0.7, 0.9, 0.9, 0.9, 0.0, 0.4, 0.6, 0.8, 0.6, 0.7, 0.6, 0.3};
    // clang-format on
    const std::size_t expected_rows = 12;
    EXPECT_EQ(expected_rows * num_cols, expected_output.size());

    DevMemInfo dm{input_buffer, dtype, {num_rows, num_cols}, {1, num_rows}};
    ShapeType output_shape{expected_rows, num_cols};
    auto output_buffer = MatxUtil::reduce_max(dm, seq_ids, 0, output_shape);

    EXPECT_EQ(output_buffer->size(), expected_rows * num_cols * dtype.item_size());

    std::vector<double> output(expected_rows * num_cols);
    MRC_CHECK_CUDA(cudaMemcpy(output.data(), output_buffer->data(), output_buffer->size(), cudaMemcpyDeviceToHost));

    EXPECT_EQ(output.size(), expected_output.size());
    for (std::size_t i = 0; i < output.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(output[i], expected_output[i]);
    }
}

TEST_F(TestMatxUtil, Cast)
{
    std::vector<float> float_vec{5.1, 2.2, 8.3, 9.4, 8.5, 2.6, 1.7, 8.1};

    DType float_type(TypeId::FLOAT32);

    auto float_buffer =
        std::make_shared<rmm::device_buffer>(float_vec.size() * float_type.item_size(), rmm::cuda_stream_per_thread);

    MRC_CHECK_CUDA(cudaMemcpy(float_buffer->data(), float_vec.data(), float_buffer->size(), cudaMemcpyHostToDevice));

    DevMemInfo dm{float_buffer, float_type, {4, 2}, {1, 4}};

    DType double_type(TypeId::FLOAT64);
    auto double_buffer = MatxUtil::cast(dm, double_type.type_id());
    EXPECT_EQ(float_vec.size() * double_type.item_size(), double_buffer->size());

    std::vector<double> double_vec(float_vec.size());
    MRC_CHECK_CUDA(cudaMemcpy(double_vec.data(), double_buffer->data(), double_buffer->size(), cudaMemcpyDeviceToHost));

    EXPECT_EQ(double_vec.size(), float_vec.size());
    for (std::size_t i = 0; i < double_vec.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(double_vec[i], float_vec[i]);
    }
}

TEST_F(TestMatxUtil, Threshold)
{
    // clang-format off
    // disabling clang-format to illustrate row-major layout

    std::vector<float> input
    {
        1.0, 0.2, 0.7, 0.9,
        1.0, 0.6, 0.1, 0.9,
        0.2, 0.8, 1.0, 0.9,
        0.1, 0.4, 0.1, 0.3,
        0.8, 1.0, 1.0, 0.8
    };

    std::vector<bool> expected_output
    {
        true,  false, true,  true,
        true,  true,  false, true,
        false, true,  true,  true,
        false, false, false, false,
        true,  true,  true,  true,
    };
    // clang-format on

    TensorIndex num_cols = 4;
    TensorIndex num_rows = 5;
    EXPECT_EQ(num_cols * num_rows, input.size());

    DType dtype(TypeId::FLOAT32);

    std::size_t buff_size = input.size() * dtype.item_size();
    auto input_buffer     = std::make_shared<rmm::device_buffer>(buff_size, rmm::cuda_stream_per_thread);

    MRC_CHECK_CUDA(cudaMemcpy(input_buffer->data(), input.data(), input_buffer->size(), cudaMemcpyHostToDevice));

    DevMemInfo dm{input_buffer, dtype, {num_rows, num_cols}, {num_cols, 1}};

    auto output = MatxUtil::threshold(dm, 0.5, false);

    // output and output_by_row are holding 1-byte bool values, so the byte size and element size should be the same
    EXPECT_EQ(output->size(), expected_output.size());

    std::vector<uint8_t> host_byte_output(expected_output.size());

    MRC_CHECK_CUDA(cudaMemcpy(host_byte_output.data(), output->data(), output->size(), cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < host_byte_output.size(); ++i)
    {
        bool output_val = host_byte_output[i];
        EXPECT_EQ(output_val, expected_output[i]);
    }
}

TEST_F(TestMatxUtil, ThresholdByRow)
{
    // clang-format off
    // disabling clang-format to illustrate row-major layout

    std::vector<float> input
    {
        1.0, 0.2, 0.7, 0.9,
        1.0, 0.6, 0.1, 0.9,
        0.2, 0.8, 1.0, 0.9,
        0.1, 0.4, 0.1, 0.3,
        0.8, 1.0, 1.0, 0.8
    };

    std::vector<bool> expected_output{true, true, true, false, true};
    // clang-format on

    TensorIndex num_cols = 4;
    TensorIndex num_rows = 5;
    EXPECT_EQ(num_cols * num_rows, input.size());

    DType dtype(TypeId::FLOAT32);

    std::size_t buff_size = input.size() * dtype.item_size();
    auto input_buffer     = std::make_shared<rmm::device_buffer>(buff_size, rmm::cuda_stream_per_thread);

    MRC_CHECK_CUDA(cudaMemcpy(input_buffer->data(), input.data(), input_buffer->size(), cudaMemcpyHostToDevice));

    DevMemInfo dm{input_buffer, dtype, {num_rows, num_cols}, {num_cols, 1}};

    auto output = MatxUtil::threshold(dm, 0.5, true);

    // output and output_by_row are holding 1-byte bool values, so the byte size and element size should be the same
    EXPECT_EQ(output->size(), expected_output.size());

    std::vector<uint8_t> host_byte_output(expected_output.size());

    MRC_CHECK_CUDA(cudaMemcpy(host_byte_output.data(), output->data(), output->size(), cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < host_byte_output.size(); ++i)
    {
        bool output_val = host_byte_output[i];
        EXPECT_EQ(output_val, expected_output[i]);
    }
}
