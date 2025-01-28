/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/export.h"              // for MORPHEUS_EXPORT
#include "morpheus/messages/control.hpp"  // for ControlMessage

#include <boost/fiber/context.hpp>                   // for operator<<
#include <cudf/strings/strings_column_view.hpp>      // for strings_column_view
#include <mrc/segment/builder.hpp>                   // for Builder
#include <mrc/segment/object.hpp>                    // for Object
#include <nvtext/subword_tokenize.hpp>               // for tokenizer_result
#include <pymrc/node.hpp>                            // for PythonNode
#include <rmm/mr/device/device_memory_resource.hpp>  // for device_memory_resource
#include <rxcpp/rx.hpp>                              // for trace_activity, decay_t, from

#include <cstdint>  // for uint32_t
#include <memory>   // for shared_ptr, allocator
#include <string>   // for string
#include <thread>   // for operator<<

// IWYU pragma: no_include "rxcpp/sources/rx-iterate.hpp"

namespace morpheus {
/****** Component public implementations *******************/
/****** PreprocessNLPStage**********************************/

/**
 * @addtogroup stages
 * @{
 * @file
 */

/**
 * @brief NLP input data for inference
 */
class MORPHEUS_EXPORT PreprocessNLPStage
  : public mrc::pymrc::PythonNode<std::shared_ptr<ControlMessage>, std::shared_ptr<ControlMessage>>
{
  public:
    using base_t = mrc::pymrc::PythonNode<std::shared_ptr<ControlMessage>, std::shared_ptr<ControlMessage>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    /**
     * @brief Construct a new Preprocess NLP Stage object
     *
     * @param vocab_hash_file : Path to hash file containing vocabulary of words with token-ids. This can be created
     * from the raw vocabulary using the `cudf.utils.hash_vocab_utils.hash_vocab` function.
     * @param sequence_length : Sequence Length to use (We add to special tokens for NER classification job).
     * @param truncation : If set to true, strings will be truncated and padded to max_length. Each input string will
     * result in exactly one output sequence. If set to false, there may be multiple output sequences when the
     * max_length is smaller than generated tokens.
     * @param do_lower_case : If set to true, original text will be lowercased before encoding.
     * @param add_special_token : Whether or not to encode the sequences with the special tokens of the BERT
     * classification model.
     * @param stride : If `truncation` == False and the tokenized string is larger than max_length, the sequences
     * containing the overflowing token-ids can contain duplicated token-ids from the main sequence. If max_length is
     * equal to stride there are no duplicated-id tokens. If stride is 80% of max_length, 20% of the first sequence will
     * be repeated on the second sequence and so on until the entire sentence is encoded.
     * @param column : Name of the string column to operate on, defaults to "data".
     */
    PreprocessNLPStage(std::string vocab_hash_file,
                       uint32_t sequence_length,
                       bool truncation,
                       bool do_lower_case,
                       bool add_special_token,
                       int stride         = -1,
                       std::string column = "data");

    /**
     * Called every time a message is passed to this stage
     */
    source_type_t on_data(sink_type_t x);

  private:
    nvtext::tokenizer_result subword_tokenize(const std::string& vocab_hash_file,
                                              uint32_t sequence_length,
                                              bool do_lower_case,
                                              bool truncation,
                                              cudf::strings_column_view const& string_col,
                                              int stride,
                                              rmm::mr::device_memory_resource* mr);
    std::string m_vocab_hash_file;
    std::string m_column;
    uint32_t m_sequence_length;
    bool m_truncation;
    bool m_do_lower_case;
    bool m_add_special_token;
    int m_stride{-1};
};

/****** PreprocessNLPStageInferenceProxy********************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MORPHEUS_EXPORT PreprocessNLPStageInterfaceProxy
{
    /**
     * @brief Create and initialize a ProcessNLPStage that receives ControlMessage and emits ControlMessage, and return
     * the result
     *
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @param vocab_hash_file : Path to hash file containing vocabulary of words with token-ids. This can be created
     * from the raw vocabulary using the `cudf.utils.hash_vocab_utils.hash_vocab` function.
     * @param sequence_length : Sequence Length to use (We add to special tokens for NER classification job).
     * @param truncation : If set to true, strings will be truncated and padded to max_length. Each input string will
     * result in exactly one output sequence. If set to false, there may be multiple output sequences when the
     * max_length is smaller than generated tokens.
     * @param do_lower_case : If set to true, original text will be lowercased before encoding.
     * @param add_special_token : Whether or not to encode the sequences with the special tokens of the BERT
     * classification model.
     * @param stride : If `truncation` == False and the tokenized string is larger than max_length, the sequences
     * containing the overflowing token-ids can contain duplicated token-ids from the main sequence. If max_length is
     * equal to stride there are no duplicated-id tokens. If stride is 80% of max_length, 20% of the first sequence will
     * be repeated on the second sequence and so on until the entire sentence is encoded.
     * @param column : Name of the string column to operate on, defaults to "data".
     * @return std::shared_ptr<mrc::segment::Object<PreprocessNLPStage>>
     */
    static std::shared_ptr<mrc::segment::Object<PreprocessNLPStage>> init(mrc::segment::Builder& builder,
                                                                          const std::string& name,
                                                                          std::string vocab_hash_file,
                                                                          uint32_t sequence_length,
                                                                          bool truncation,
                                                                          bool do_lower_case,
                                                                          bool add_special_token,
                                                                          int stride         = -1,
                                                                          std::string column = "data");
};
/** @} */  // end of group
}  // namespace morpheus
