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

#include "morpheus/types.hpp"  // For TensorIndex, TensorSize
#include "morpheus/utilities/matx_util.hpp"

#include <boost/numeric/conversion/cast.hpp>  // for numeric_cast
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <matx.h>
#include <mrc/cuda/sync.hpp>

#include <array>
#include <cstddef>  // for size_t

namespace {
using namespace morpheus;
using tensorShape_1d = std::array<matx::index_t, 1>;
using tensorShape_2d = std::array<matx::index_t, 2>;

// Since we are building MatX in 32bit mode, we can only support up to 2^31 in any on dimension, for count type values
// that consider multiple dimensions we use TensorSize, while other operations such as MatxUtil__MatxCast which only
// opperate on a single dimension use TensorIndex.

// Component-private classes.
// ************ MatxUtil__MatxCast**************//
/**
 * TODO(Documentation)
 */
struct MatxUtil__MatxCast
{
    TensorIndex element_count;
    rmm::cuda_stream_view stream;

    /**
     * TODO(Documentation)
     */
    template <typename InputT,
              typename OutputT,
              std::enable_if_t<!cudf::is_numeric<InputT>() || !cudf::is_numeric<OutputT>()>* = nullptr>
    void operator()(void* input_data, void* output_data)
    {
        throw std::invalid_argument("Unsupported conversion");
    }

    /**
     * TODO(Documentation)
     */
    template <typename InputT,
              typename OutputT,
              std::enable_if_t<cudf::is_numeric<InputT>() && cudf::is_numeric<OutputT>()>* = nullptr>
    void operator()(void* input_data, void* output_data)
    {
        tensorShape_1d shape({element_count});

        auto input_tensor  = matx::make_tensor<InputT>(static_cast<InputT*>(input_data), shape);
        auto output_tensor = matx::make_tensor<OutputT>(static_cast<OutputT*>(output_data), shape);

        (output_tensor = input_tensor).run(stream.value());
    }
};

// ************ MatxUtil__MatxCreateSegIds**************//
/**
 * TODO(Documentation)
 */
struct MatxUtil__MatxCreateSegIds
{
    TensorIndex start_idx;
    TensorIndex element_count;
    TensorIndex fea_len;
    rmm::cuda_stream_view stream;

    /**
     * TODO(Documentation)
     */
    template <typename OutputT, std::enable_if_t<!std::is_integral_v<OutputT>>* = nullptr>
    void operator()(void* output_data)
    {
        throw std::invalid_argument("Unsupported conversion");
    }

    /**
     * TODO(Documentation)
     */
    template <typename OutputT, std::enable_if_t<std::is_integral_v<OutputT>>* = nullptr>
    void operator()(void* output_data)
    {
        tensorShape_2d shape({element_count, 3});

        auto output_tensor = matx::make_tensor<OutputT>(static_cast<OutputT*>(output_data), shape);

        auto col0      = output_tensor.template Slice<1>({0, 0}, {matx::matxEnd, matx::matxDropDim});
        auto col1      = output_tensor.template Slice<1>({0, 1}, {matx::matxEnd, matx::matxDropDim});
        auto col2      = output_tensor.template Slice<1>({0, 2}, {matx::matxEnd, matx::matxDropDim});
        auto range_col = matx::range<0, tensorShape_1d, OutputT>({element_count}, start_idx, 1);

        (col0 = range_col).run(stream.value());
        (col1 = 0).run(stream.value());
        (col2 = fea_len - 1).run(stream.value());
    }
};

// ************ MatxUtil__MatxOffsetSegIds**************//
/**
 * TODO(Documentation)
 */
struct MatxUtil__MatxOffsetSegIds
{
    TensorIndex offset;
    TensorIndex element_count;
    rmm::cuda_stream_view stream;

    /**
     * TODO(Documentation)
     */
    template <typename InputT, std::enable_if_t<!std::is_integral_v<InputT>>* = nullptr>
    void operator()(void* output_data)
    {
        throw std::invalid_argument("Unsupported conversion");
    }

    /**
     * TODO(Documentation)
     */
    template <typename InputT, std::enable_if_t<std::is_integral_v<InputT>>* = nullptr>
    void operator()(void* input_data)
    {
        tensorShape_2d shape({element_count, 3});

        auto input_tensor = matx::make_tensor<InputT>(static_cast<InputT*>(input_data), shape);

        auto col0 = input_tensor.template Slice<1>({0, 0}, {matx::matxEnd, matx::matxDropDim});

        // Simply add the offset to the column
        (col0 = col0 + offset).run(stream.value());
    }
};

// ************ MatxUtil__MatxLogits**************//
/**
 * TODO(Documentation)
 */
struct MatxUtil__MatxLogits
{
    TensorIndex element_count;
    rmm::cuda_stream_view stream;

    /**
     * TODO(Documentation)
     */
    template <typename InputT, std::enable_if_t<!cudf::is_floating_point<InputT>()>* = nullptr>
    void operator()(void* input_data, void* output_data)
    {
        throw std::invalid_argument("Unsupported conversion");
    }

    /**
     * TODO(Documentation)
     */
    template <typename InputT, std::enable_if_t<cudf::is_floating_point<InputT>()>* = nullptr>
    void operator()(void* input_data, void* output_data)
    {
        tensorShape_1d shape({element_count});

        auto input_tensor = matx::make_tensor<InputT>(static_cast<InputT*>(input_data), shape);

        auto output_tensor = matx::make_tensor<InputT>(static_cast<InputT*>(output_data), shape);

        (output_tensor = (InputT)1 / ((InputT)1 + matx::exp((InputT)-1 * input_tensor))).run(stream.value());
    }
};

// ************ MatxUtil__MatxTranspose**************//
/**
 * TODO(Documentation)
 */
struct MatxUtil__MatxTranspose
{
    rmm::cuda_stream_view stream;
    TensorIndex rows;
    TensorIndex cols;

    /**
     * TODO(Documentation)
     */
    template <typename InputT, std::enable_if_t<!cudf::is_numeric<InputT>()>* = nullptr>
    void operator()(void* input_data, void* output_data)
    {
        throw std::invalid_argument("Unsupported conversion");
    }

    /**
     * TODO(Documentation)
     */
    template <typename InputT, std::enable_if_t<cudf::is_numeric<InputT>()>* = nullptr>
    void operator()(void* input_data, void* output_data)
    {
        tensorShape_2d input_shape({rows, cols});
        tensorShape_2d output_shape({cols, rows});

        auto input_tensor  = matx::make_tensor<InputT>(static_cast<InputT*>(input_data), input_shape);
        auto output_tensor = matx::make_tensor<InputT>(static_cast<InputT*>(output_data), output_shape);

        (output_tensor = input_tensor.Permute({1, 0})).run(stream.value());
    }
};

// ************ MatxUtil__MatxThreshold**************//
/**
 * TODO(Documentation)
 */
struct MatxUtil__MatxThreshold
{
    TensorIndex rows;
    TensorIndex cols;
    bool by_row;
    rmm::cuda_stream_view stream;

    /**
     * TODO(Documentation)
     */
    template <typename InputT, std::enable_if_t<!cudf::is_floating_point<InputT>()>* = nullptr>
    void operator()(void* input_data, void* output_data, double threshold, const ShapeType& stride)
    {
        throw std::invalid_argument("Unsupported conversion");
    }

    /**
     * TODO(Documentation)
     */
    template <typename InputT, std::enable_if_t<cudf::is_floating_point<InputT>()>* = nullptr>
    void operator()(void* input_data, void* output_data, double threshold, const ShapeType& stride)
    {
        if (by_row)
        {
            this->threshold_by_row<InputT>(input_data, output_data, threshold, stride);
        }
        else
        {
            this->threshold<InputT>(input_data, output_data, threshold, stride);
        }
    }

  private:
    /**
     * TODO(Documentation)
     */
    template <typename InputT>
    void threshold_by_row(void* input_data, void* output_data, double threshold, const ShapeType& stride)
    {
        // Output is always 1 column
        tensorShape_1d output_shape({rows});

        matx::DefaultDescriptor<2> desc{{rows, cols}, {stride[0], stride[1]}};

        auto input_tensor =
            matx::make_tensor<InputT, matx::DefaultDescriptor<2>>(static_cast<InputT*>(input_data), std::move(desc));

        auto output_tensor = matx::make_tensor<bool>(static_cast<bool*>(output_data), output_shape);

        // Convert max value to bool
        (output_tensor = matx::max(input_tensor, {1}) > (InputT)threshold).run(stream.value());
    }

    /**
     * TODO(Documentation)
     */
    template <typename InputT>
    void threshold(void* input_data, void* output_data, double threshold, const ShapeType& stride)
    {
        matx::DefaultDescriptor<2> input_desc{{rows, cols}, {stride[0], stride[1]}};

        // Input & Output have the same shape & stride. The make_tensor API requires a move for the descriptor
        // so we need to take a copy of it here.
        matx::DefaultDescriptor<2> output_desc = input_desc;

        auto input_tensor  = matx::make_tensor<InputT>(static_cast<InputT*>(input_data), std::move(input_desc));
        auto output_tensor = matx::make_tensor<bool>(static_cast<bool*>(output_data), std::move(output_desc));

        // Convert max value to bool
        (output_tensor = input_tensor > (InputT)threshold).run(stream.value());
    }
};

struct MatxUtil__MatxReduceMax
{
    matx::index_t num_input_rows;
    matx::index_t num_output_rows;
    matx::index_t num_cols;
    std::vector<matx::index_t> input_stride;
    const ShapeType& seq_ids;
    TensorIndex seq_id_offset;
    rmm::cuda_stream_view stream;

    template <typename InputT, std::enable_if_t<!cudf::is_floating_point<InputT>()>* = nullptr>
    void operator()(void* input_data, void* output_data)
    {
        throw std::invalid_argument("Unsupported conversion");
    }

    template <typename InputT, std::enable_if_t<cudf::is_floating_point<InputT>()>* = nullptr>
    void operator()(void* input_data, void* output_data)
    {
        auto input_ptr = static_cast<InputT*>(input_data);
        matx::DefaultDescriptor<2> input_desc{{num_input_rows, num_cols}, {input_stride[0], input_stride[1]}};
        auto input_tensor = matx::make_tensor<InputT, matx::DefaultDescriptor<2>>(input_ptr, std::move(input_desc));

        auto output_ptr = static_cast<InputT*>(output_data);

        matx::index_t output_stride[2] = {input_stride[0], input_stride[1]};
        if (output_stride[0] == 1)
        {
            output_stride[1] = num_output_rows;
        }

        matx::DefaultDescriptor<2> output_desc{{num_output_rows, num_cols}, output_stride};
        auto output_tensor = matx::make_tensor<InputT, matx::DefaultDescriptor<2>>(output_ptr, std::move(output_desc));

        matx::index_t start = 0;
        auto output_offset  = seq_ids[seq_id_offset];
        for (matx::index_t i = 1; i < num_input_rows; ++i)
        {
            auto idx = seq_ids[i + seq_id_offset];
            if (idx != seq_ids[start + seq_id_offset])
            {
                DCHECK(seq_ids[start + seq_id_offset] - output_offset < num_output_rows);
                reduce_rows(input_tensor, output_tensor, start, i, seq_ids[start + seq_id_offset] - output_offset);
                start = i;
            }
        }

        DCHECK(seq_ids[start + seq_id_offset] - output_offset < num_output_rows)
            << "\nstart=" << start
            << " seq_ids[start+seq_id_offset]-output_offset=" << seq_ids[start + seq_id_offset] - output_offset
            << " num_output_rows=" << num_output_rows;
        reduce_rows(input_tensor, output_tensor, start, num_input_rows, seq_ids[start + seq_id_offset] - output_offset);
    }

    template <typename InputT>
    void reduce_rows(matx::tensor_t<InputT, 2>& input_tensor,
                     matx::tensor_t<InputT, 2>& output_tensor,
                     matx::index_t start,
                     matx::index_t stop,
                     matx::index_t output_idx)
    {
        auto input_slice = input_tensor.Slice({start, 0}, {stop, matx::matxEnd});

        auto output_slice = output_tensor.template Slice<1>({output_idx, 0}, {matx::matxDropDim, matx::matxEnd});

        (output_slice = matx::max(input_slice.Permute({1, 0}))).run(stream.value());
    }
};
}  // namespace

namespace morpheus {
// Component public implementations
// ************ MatxUtil************************* //
std::shared_ptr<rmm::device_buffer> MatxUtil::cast(const DevMemInfo& input, TypeId output_type)
{
    auto output_dtype = DType(output_type);

    // Create the output
    auto output = input.make_new_buffer(output_dtype.item_size() * input.count());

    cudf::double_type_dispatcher(cudf::data_type{input.dtype().cudf_type_id()},
                                 cudf::data_type{output_dtype.cudf_type_id()},
                                 MatxUtil__MatxCast{boost::numeric_cast<TensorIndex>(input.count()), output->stream()},
                                 input.data(),
                                 output->data());

    mrc::enqueue_stream_sync_event(output->stream()).get();

    return output;
}

std::shared_ptr<rmm::device_buffer> MatxUtil::create_seq_ids(TensorIndex row_count,
                                                             TensorIndex fea_len,
                                                             TypeId output_type,
                                                             std::shared_ptr<MemoryDescriptor> md,
                                                             TensorIndex start_idx)
{
    auto output_dtype = DType(output_type);

    // Now create the output
    auto output = std::make_shared<rmm::device_buffer>(
        output_dtype.item_size() * row_count * 3, md->cuda_stream, md->memory_resource);

    cudf::type_dispatcher(cudf::data_type{output_dtype.cudf_type_id()},
                          MatxUtil__MatxCreateSegIds{start_idx, row_count, fea_len, output->stream()},
                          output->data());

    return output;
}

void MatxUtil::offset_seq_ids(const DevMemInfo& input, TensorIndex offset)
{
    cudf::type_dispatcher(cudf::data_type{input.dtype().cudf_type_id()},
                          MatxUtil__MatxOffsetSegIds{offset, input.shape(0), rmm::cuda_stream_per_thread},
                          input.data());

    mrc::enqueue_stream_sync_event(rmm::cuda_stream_per_thread).get();
}

std::shared_ptr<rmm::device_buffer> MatxUtil::logits(const DevMemInfo& input)
{
    // Create the output
    auto output = input.make_new_buffer(input.bytes());

    cudf::type_dispatcher(cudf::data_type{input.dtype().cudf_type_id()},
                          MatxUtil__MatxLogits{boost::numeric_cast<TensorIndex>(input.count()), output->stream()},
                          input.data(),
                          output->data());

    return output;
}

std::shared_ptr<rmm::device_buffer> MatxUtil::transpose(const DevMemInfo& input)
{
    // Now create the output
    auto output = input.make_new_buffer(input.bytes());

    cudf::type_dispatcher(cudf::data_type{input.dtype().cudf_type_id()},
                          MatxUtil__MatxTranspose{output->stream(), input.shape(0), input.shape(1)},
                          input.data(),
                          output->data());

    return output;
}

std::shared_ptr<rmm::device_buffer> MatxUtil::threshold(const DevMemInfo& input, double thresh_val, bool by_row)
{
    const auto rows        = input.shape(0);
    const auto cols        = input.shape(1);
    TensorSize output_size = sizeof(bool) * rows;
    if (!by_row)
    {
        output_size *= cols;
    }

    // Now create the output array of bools
    auto output = input.make_new_buffer(output_size);

    cudf::type_dispatcher(cudf::data_type{input.dtype().cudf_type_id()},
                          MatxUtil__MatxThreshold{rows, cols, by_row, output->stream()},
                          input.data(),
                          output->data(),
                          thresh_val,
                          input.stride());

    mrc::enqueue_stream_sync_event(output->stream()).get();

    return output;
}

std::shared_ptr<rmm::device_buffer> MatxUtil::reduce_max(const DevMemInfo& input,
                                                         const ShapeType& seq_ids,
                                                         TensorIndex seq_id_offset,
                                                         const ShapeType& output_shape)
{
    const auto& dtype   = input.dtype();
    auto cudf_type      = cudf::data_type{dtype.cudf_type_id()};
    auto num_input_rows = input.shape(0);
    auto num_input_cols = input.shape(1);

    TensorSize output_element_count = output_shape[0] * output_shape[1];
    TensorSize output_buff_size     = dtype.item_size() * output_element_count;

    DCHECK(output_element_count <= input.count()) << "Output buffer size should be less than or equal to the input";
    DCHECK(num_input_cols == output_shape[1]) << "Number of input and output columns must match";

    auto output = input.make_new_buffer(output_buff_size);

    MatxUtil__MatxReduceMax matx_reduce_max{
        num_input_rows, output_shape[0], num_input_cols, input.stride(), seq_ids, seq_id_offset, output->stream()};

    cudf::type_dispatcher(cudf_type, matx_reduce_max, input.data(), output->data());

    mrc::enqueue_stream_sync_event(output->stream()).get();
    return output;
}
}  // namespace morpheus
