/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "../test_utils/common.hpp"  // for get_morpheus_root, TestWithPythonInterpreter, morpheus

#include "morpheus/io/deserializers.hpp"               // for load_table_from_file
#include "morpheus/messages/control.hpp"               // for ControlMessage
#include "morpheus/messages/memory/tensor_memory.hpp"  // for TensorMemory
#include "morpheus/messages/meta.hpp"                  // for MessageMeta
#include "morpheus/objects/tensor_object.hpp"          // for TensorObject
#include "morpheus/stages/preprocess_nlp.hpp"          // for PreprocessNLPStage, PreprocessNLPStageCC, PreprocessNL...
#include "morpheus/types.hpp"                          // for TensorIndex

#include <cuda_runtime.h>       // for cudaMemcpy, cudaMemcpyKind
#include <gtest/gtest.h>        // for EXPECT_EQ, Message, TestPartResult, TestInfo, TEST_F
#include <mrc/cuda/common.hpp>  // for __check_cuda_errors, MRC_CHECK_CUDA
#include <pybind11/gil.h>       // for gil_scoped_acquire, gil_scoped_release

#include <cstdint>     // for int32_t
#include <filesystem>  // for operator/, path
#include <memory>      // for allocator, make_shared, __shared_ptr_access, shared_ptr
#include <utility>     // for move
#include <vector>      // for vector

using namespace morpheus;

TEST_CLASS_WITH_PYTHON(PreprocessNLP);

TEST_F(TestPreprocessNLP, TestProcessControlMessage)
{
    pybind11::gil_scoped_release no_gil;
    auto test_data_dir               = test::get_morpheus_root() / "tests/tests_data";
    std::filesystem::path input_file = test_data_dir / "countries_sample.csv";

    auto test_vocab_hash_file_dir         = test::get_morpheus_root() / "python/morpheus/morpheus/data";
    std::filesystem::path vocab_hash_file = test_vocab_hash_file_dir / "bert-base-cased-hash.txt";

    // Create a dataframe from a file
    auto table = load_table_from_file(input_file);
    auto meta  = MessageMeta::create_from_cpp(std::move(table));

    // Create ControlMessage
    auto cm = std::make_shared<ControlMessage>();
    cm->payload(meta);

    // Create PreProcessControlMessageStage
    auto cm_stage = std::make_shared<PreprocessNLPStage>(vocab_hash_file /*vocab_hash_file*/,
                                                         1 /*sequence_length*/,
                                                         false /*truncation*/,
                                                         false /*do_lower_case*/,
                                                         false /*add_special_token*/,
                                                         1 /*stride*/,
                                                         "country" /*column*/);

    auto cm_response = cm_stage->on_data(cm);
    auto cm_tensors  = cm_response->tensors();

    // Verify output tensors
    std::vector<int32_t> expected_input_ids = {6469, 10278, 11347, 1262, 27583, 13833};
    auto cm_input_ids                       = cm_tensors->get_tensor("input_ids");
    std::vector<int32_t> cm_input_ids_host(cm_input_ids.count());
    MRC_CHECK_CUDA(cudaMemcpy(
        cm_input_ids_host.data(), cm_input_ids.data(), cm_input_ids.count() * sizeof(int32_t), cudaMemcpyDeviceToHost));

    EXPECT_EQ(expected_input_ids, cm_input_ids_host);

    std::vector<int32_t> expected_input_mask = {1, 1, 1, 1, 1, 1};
    auto cm_input_mask                       = cm_tensors->get_tensor("input_mask");

    std::vector<int32_t> cm_input_mask_host(cm_input_mask.count());

    MRC_CHECK_CUDA(cudaMemcpy(cm_input_mask_host.data(),
                              cm_input_mask.data(),
                              cm_input_mask.count() * sizeof(int32_t),
                              cudaMemcpyDeviceToHost));

    EXPECT_EQ(expected_input_mask, cm_input_mask_host);

    std::vector<int32_t> expected_seq_ids = {0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 3, 0, 0, 4, 0, 0};
    auto cm_seq_ids                       = cm_tensors->get_tensor("seq_ids");

    std::vector<TensorIndex> cm_seq_ids_host(cm_seq_ids.count());

    MRC_CHECK_CUDA(cudaMemcpy(
        cm_seq_ids_host.data(), cm_seq_ids.data(), cm_seq_ids.count() * sizeof(TensorIndex), cudaMemcpyDeviceToHost));

    EXPECT_EQ(expected_seq_ids, cm_seq_ids_host);
}
