/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "morpheus/stages/preprocess_nlp.hpp"

#include "mrc/segment/object.hpp"  // for Object

#include "morpheus/messages/control.hpp"               // for ControlMessage
#include "morpheus/messages/memory/tensor_memory.hpp"  // for TensorMemory
#include "morpheus/messages/meta.hpp"                  // for MessageMeta
#include "morpheus/objects/dtype.hpp"                  // for DType
#include "morpheus/objects/table_info.hpp"             // for TableInfo
#include "morpheus/objects/tensor.hpp"                 // for Tensor
#include "morpheus/types.hpp"                          // for TensorIndex

#include <cudf/column/column.hpp>                 // for column
#include <cudf/column/column_factories.hpp>       // for make_column_from_scalar
#include <cudf/column/column_view.hpp>            // for column_view
#include <cudf/filling.hpp>                       // for sequence
#include <cudf/reshape.hpp>                       // for interleave_columns
#include <cudf/scalar/scalar.hpp>                 // for numeric_scalar
#include <cudf/strings/strings_column_view.hpp>   // for strings_column_view
#include <cudf/table/table_view.hpp>              // for table_view
#include <cudf/types.hpp>                         // for type_id, data_type
#include <cudf/unary.hpp>                         // for cast
#include <mrc/segment/builder.hpp>                // for Builder
#include <nvtext/normalize.hpp>                   // for normalize_spaces
#include <nvtext/subword_tokenize.hpp>            // for tokenizer_result, load_vocabulary_file, subword_tokenize
#include <rmm/cuda_stream_view.hpp>               // for cuda_stream_default
#include <rmm/device_buffer.hpp>                  // for device_buffer
#include <rmm/mr/device/per_device_resource.hpp>  // for get_current_device_resource

#include <cstdint>  // for uint32_t, int32_t
#include <memory>   // for shared_ptr, unique_ptr, __shared_ptr_access, allocator
#include <utility>  // for move
#include <vector>   // for vector

namespace morpheus {
// Component public implementations
// ************ PreprocessNLPStage ************************* //
PreprocessNLPStage::PreprocessNLPStage(std::string vocab_hash_file,
                                       uint32_t sequence_length,
                                       bool truncation,
                                       bool do_lower_case,
                                       bool add_special_token,
                                       int stride,
                                       std::string column) :
  base_t(rxcpp::operators::map([this](sink_type_t x) {
      return this->on_data(std::move(x));
  })),
  m_vocab_hash_file(std::move(vocab_hash_file)),
  m_sequence_length(sequence_length),
  m_truncation(truncation),
  m_do_lower_case(do_lower_case),
  m_add_special_token(add_special_token),
  m_column(std::move(column))
{
    // Auto calc stride to be 75% of sequence length
    if (stride < 0)
    {
        stride = m_sequence_length / 2;
        stride = stride + stride / 2;
    }

    m_stride = stride;
}

PreprocessNLPStage::source_type_t PreprocessNLPStage::on_data(sink_type_t msg)
{
    // Convert to string view
    auto meta = msg->payload()->get_info(this->m_column);

    auto col        = meta.get_column(0);
    auto string_col = cudf::strings_column_view{col};

    auto token_results = subword_tokenize(this->m_vocab_hash_file,
                                          this->m_sequence_length,
                                          this->m_do_lower_case,
                                          this->m_truncation,
                                          string_col,
                                          this->m_stride,
                                          rmm::mr::get_current_device_resource());

    // Build the results
    auto memory = std::make_shared<TensorMemory>(token_results.nrows_tensor);

    TensorIndex length = token_results.tensor_token_ids->size() / token_results.sequence_length;
    auto input_ids_released =
        cudf::cast(token_results.tensor_token_ids->view(), cudf::data_type(cudf::type_id::INT32))->release();
    memory->set_tensor("input_ids",
                       Tensor::create(std::move(input_ids_released.data),
                                      DType::create<int32_t>(),
                                      {length, static_cast<TensorIndex>(token_results.sequence_length)},
                                      {},
                                      0));

    length = token_results.tensor_attention_mask->size() / token_results.sequence_length;
    auto input_mask_released =
        cudf::cast(token_results.tensor_attention_mask->view(), cudf::data_type(cudf::type_id::INT32))->release();
    memory->set_tensor("input_mask",
                       Tensor::create(std::move(input_mask_released.data),
                                      DType::create<int32_t>(),
                                      {length, static_cast<TensorIndex>(token_results.sequence_length)},
                                      {},
                                      0));

    auto tensor_index_dtype = DType::create<TensorIndex>();
    length                  = token_results.tensor_metadata->size() / 3;
    auto seq_ids_released =
        cudf::cast(token_results.tensor_metadata->view(), cudf::data_type(tensor_index_dtype.cudf_type_id()))
            ->release();

    std::shared_ptr<rmm::device_buffer> seq_ids_data = std::move(seq_ids_released.data);

    memory->set_tensor("seq_ids", Tensor::create(seq_ids_data, tensor_index_dtype, {length, 3}, {}, 0));

    auto next = msg;
    next->tensors(memory);

    return std::move(next);
}

nvtext::tokenizer_result PreprocessNLPStage::subword_tokenize(const std::string& vocab_hash_file,
                                                              uint32_t sequence_length,
                                                              bool do_lower_case,
                                                              bool truncation,
                                                              cudf::strings_column_view const& string_col,
                                                              int stride,
                                                              rmm::mr::device_memory_resource* mr)
{
    // Create the hashed vocab
    thread_local std::unique_ptr<nvtext::hashed_vocabulary> vocab = nvtext::load_vocabulary_file(vocab_hash_file);

    // remove leading and trailing whitespace
    auto normalized_col      = nvtext::normalize_spaces(string_col);
    auto normalized_col_view = cudf::strings_column_view{normalized_col->view()};

    // Perform the tokenizer
    nvtext::tokenizer_result token_results;

    if (normalized_col_view.chars_size(rmm::cuda_stream_default) > 0)
    {
        token_results = nvtext::subword_tokenize(normalized_col_view,
                                                 *vocab,
                                                 sequence_length,
                                                 stride,
                                                 do_lower_case,
                                                 truncation,
                                                 rmm::cuda_stream_default,
                                                 mr);
    }
    else
    {
        // workaround for a situation where the input strings contain either no
        // characters or only whitespace
        auto zero     = cudf::numeric_scalar<uint32_t>(0, true, rmm::cuda_stream_default);
        auto ids      = cudf::make_column_from_scalar(zero, sequence_length * normalized_col_view.size());
        auto mask     = cudf::make_column_from_scalar(zero, sequence_length * normalized_col_view.size());
        auto metadata = [&]() {
            auto iota   = cudf::sequence(normalized_col_view.size(), zero);
            auto zeroes = cudf::make_column_from_scalar(zero, normalized_col_view.size());
            return cudf::interleave_columns(
                cudf::table_view{std::vector<cudf::column_view>{iota->view(), zeroes->view(), zeroes->view()}});
        }();

        token_results = nvtext::tokenizer_result{static_cast<uint32_t>(normalized_col_view.size()),
                                                 sequence_length,
                                                 std::move(ids),
                                                 std::move(mask),
                                                 std::move(metadata)};
    }
    return token_results;
}

// ************ PreprocessNLPStageInterfaceProxy *********** //
std::shared_ptr<mrc::segment::Object<PreprocessNLPStage>> PreprocessNLPStageInterfaceProxy::init(
    mrc::segment::Builder& builder,
    const std::string& name,
    std::string vocab_hash_file,
    uint32_t sequence_length,
    bool truncation,
    bool do_lower_case,
    bool add_special_token,
    int stride,
    std::string column)
{
    auto stage = builder.construct_object<PreprocessNLPStage>(
        name, vocab_hash_file, sequence_length, truncation, do_lower_case, add_special_token, stride, column);

    return stage;
}
}  // namespace morpheus
