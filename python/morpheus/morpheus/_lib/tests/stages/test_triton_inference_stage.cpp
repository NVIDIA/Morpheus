/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../test_utils/common.hpp"  // for TestWithPythonInterpreter, morpheus

#include "morpheus/messages/control.hpp"               // for ControlMessage
#include "morpheus/messages/memory/tensor_memory.hpp"  // for TensorMemory
#include "morpheus/messages/meta.hpp"                  // for MessageMeta
#include "morpheus/objects/dtype.hpp"                  // for TypeId, DType
#include "morpheus/objects/memory_descriptor.hpp"      // for MemoryDescriptor
#include "morpheus/objects/tensor.hpp"                 // for Tensor
#include "morpheus/objects/tensor_object.hpp"          // for TensorObject
#include "morpheus/stages/inference_client_stage.hpp"  // for TensorModelMapping, InferenceClientStage, IInferenceCl...
#include "morpheus/stages/triton_inference.hpp"        // for TritonInferenceClient, TritonInferInput, TritonInferRe...
#include "morpheus/types.hpp"                          // for TensorMap
#include "morpheus/utilities/matx_util.hpp"            // for MatxUtil

#include <cuda_runtime.h>                         // for cudaMemcpy, cudaMemcpyKind
#include <cudf/column/column.hpp>                 // for column
#include <cudf/column/column_factories.hpp>       // for make_fixed_width_column
#include <cudf/column/column_view.hpp>            // for mutable_column_view
#include <cudf/io/types.hpp>                      // for column_name_info, table_with_metadata, table_metadata
#include <cudf/table/table.hpp>                   // for table
#include <cudf/types.hpp>                         // for data_type
#include <cudf/utilities/type_dispatcher.hpp>     // for type_to_id
#include <glog/logging.h>                         // for COMPACT_GOOGLE_LOG_INFO, DVLOG, LogMessage
#include <gtest/gtest.h>                          // for Message, TestPartResult, TestInfo, ASSERT_EQ, ASSERT_N...
#include <http_client.h>                          // for Error, InferOptions, InferenceServerHttpClient, InferR...
#include <mrc/coroutines/task.hpp>                // for Task
#include <mrc/coroutines/test_scheduler.hpp>      // for TestScheduler
#include <rmm/cuda_stream_view.hpp>               // for cuda_stream_per_thread
#include <rmm/device_buffer.hpp>                  // for device_buffer
#include <rmm/mr/device/per_device_resource.hpp>  // for get_current_device_resource

#include <cstddef>           // for size_t
#include <cstdint>           // for int32_t, int64_t, uint8_t, uint32_t
#include <functional>        // for function
#include <initializer_list>  // for initializer_list
#include <map>               // for map
#include <memory>            // for shared_ptr, allocator, unique_ptr, make_shared, make_u...
#include <numeric>           // for iota
#include <ostream>           // for char_traits, operator<<, basic_ostream
#include <stdexcept>         // for runtime_error, invalid_argument
#include <string>            // for string, operator<=>, operator<<, basic_string, char_tr...
#include <unordered_map>     // for unordered_map
#include <utility>           // for move
#include <vector>            // for vector
class FakeInferResult : public triton::client::InferResult
{
  private:
    std::map<std::string, std::vector<int32_t>> m_output_values;

  public:
    FakeInferResult(std::map<std::string, std::vector<int32_t>> output_values) :
      m_output_values(std::move(output_values))
    {}

    triton::client::Error RequestStatus() const override
    {
        throw std::runtime_error("RequestStatus not implemented");
    }

    std::string DebugString() const override
    {
        throw std::runtime_error("DebugString not implemented");
    }

    triton::client::Error Id(std::string* id) const override
    {
        throw std::runtime_error("Id not implemented");
    }

    triton::client::Error ModelName(std::string* name) const override
    {
        throw std::runtime_error("ModelName not implemented");
    }

    triton::client::Error ModelVersion(std::string* version) const override
    {
        throw std::runtime_error("ModelVersion not implemented");
    }

    triton::client::Error Shape(const std::string& output_name, std::vector<int64_t>* shape) const override
    {
        shape = new std::vector<int64_t>({0, 0});  // this is technically a leak

        return triton::client::Error::Success;
    }

    triton::client::Error Datatype(const std::string& output_name, std::string* datatype) const override
    {
        throw std::runtime_error("Datatype not implemented");
    }

    triton::client::Error StringData(const std::string& output_name,
                                     std::vector<std::string>* string_result) const override
    {
        throw std::runtime_error("StringData not implemented");
    }

    triton::client::Error RawData(const std::string& output_name, const uint8_t** buf, size_t* byte_size) const override
    {
        auto& output = m_output_values.at(output_name);
        *byte_size   = output.size() * sizeof(int32_t);
        *buf         = reinterpret_cast<uint8_t*>(const_cast<int32_t*>(output.data()));
        return triton::client::Error::Success;
    }
};

class FakeTritonClient : public morpheus::ITritonClient
{
  public:
    triton::client::Error is_server_live(bool* live) override
    {
        *live = true;
        return triton::client::Error::Success;
    }

    triton::client::Error is_server_ready(bool* ready) override
    {
        *ready = true;
        return triton::client::Error::Success;
    }

    triton::client::Error is_model_ready(bool* ready, std::string& model_name) override
    {
        *ready = true;
        return triton::client::Error::Success;
    }

    triton::client::Error model_config(std::string* model_config, std::string& model_name) override
    {
        *model_config = R"({
            "max_batch_size": 100
        })";

        return triton::client::Error::Success;
    }

    triton::client::Error model_metadata(std::string* model_metadata, std::string& model_name) override
    {
        *model_metadata = R"({
            "inputs":[
                {
                    "name":"seq_ids",
                    "shape": [0, 1],
                    "datatype":"INT32"
                }
            ],
            "outputs":[
                {
                    "name":"seq_ids",
                    "shape": [0, 1],
                    "datatype":"INT32"
                }
            ]})";

        return triton::client::Error::Success;
    }

    triton::client::Error async_infer(triton::client::InferenceServerHttpClient::OnCompleteFn callback,
                                      const triton::client::InferOptions& options,
                                      const std::vector<morpheus::TritonInferInput>& inputs,
                                      const std::vector<morpheus::TritonInferRequestedOutput>& outputs) override
    {
        callback(new FakeInferResult({{"seq_ids", std::vector<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})}}));

        return triton::client::Error::Success;
    }
};

class ErrorProneTritonClient : public FakeTritonClient
{
  private:
    bool m_is_server_live_has_errored  = false;
    bool m_is_server_live              = false;
    bool m_is_server_ready_has_errored = false;
    bool m_is_server_ready             = false;
    bool m_is_model_ready_has_errored  = false;
    bool m_is_model_ready              = false;
    bool m_model_config_has_errored    = false;
    bool m_model_metadata_has_errored  = false;
    bool m_async_infer_has_errored     = false;

  public:
    triton::client::Error is_server_live(bool* live) override
    {
        if (not m_is_server_live_has_errored)
        {
            m_is_server_live_has_errored = true;
            return triton::client::Error("is_server_live error");
        }

        *live = m_is_server_live;

        if (not m_is_server_live)
        {
            m_is_server_live = true;
        }

        return FakeTritonClient::is_server_live(live);
    }

    triton::client::Error is_server_ready(bool* ready) override
    {
        if (not m_is_server_ready_has_errored)
        {
            m_is_server_ready_has_errored = true;
            return triton::client::Error("is_server_ready error");
        }

        *ready = m_is_server_live;

        if (not m_is_server_ready)
        {
            m_is_server_ready = true;
        }

        return triton::client::Error::Success;
    }

    triton::client::Error is_model_ready(bool* ready, std::string& model_name) override
    {
        if (not m_is_model_ready_has_errored)
        {
            m_is_model_ready_has_errored = true;
            return triton::client::Error("is_model_ready error");
        }

        *ready = m_is_model_ready;

        if (not m_is_model_ready)
        {
            m_is_model_ready = true;
        }

        return triton::client::Error::Success;
    }

    triton::client::Error model_config(std::string* model_config, std::string& model_name) override
    {
        if (not m_model_config_has_errored)
        {
            m_model_config_has_errored = true;
            return triton::client::Error("model_config error");
        }

        return FakeTritonClient::model_config(model_config, model_name);
    }

    triton::client::Error model_metadata(std::string* model_metadata, std::string& model_name) override
    {
        if (not m_model_metadata_has_errored)
        {
            m_model_metadata_has_errored = true;
            return triton::client::Error("model_metadata error");
        }

        return FakeTritonClient::model_metadata(model_metadata, model_name);
    }

    triton::client::Error async_infer(triton::client::InferenceServerHttpClient::OnCompleteFn callback,
                                      const triton::client::InferOptions& options,
                                      const std::vector<morpheus::TritonInferInput>& inputs,
                                      const std::vector<morpheus::TritonInferRequestedOutput>& outputs) override
    {
        if (not m_async_infer_has_errored)
        {
            m_async_infer_has_errored = true;
            return triton::client::Error("async_infer error");
        }

        return FakeTritonClient::async_infer(callback, options, inputs, outputs);
    }
};

class TestTritonInferenceStage : public morpheus::test::TestWithPythonInterpreter
{};

cudf::io::table_with_metadata create_test_table_with_metadata(uint32_t rows)
{
    cudf::data_type cudf_data_type{cudf::type_to_id<int>()};

    auto column = cudf::make_fixed_width_column(cudf_data_type, rows);

    std::vector<int> data(rows);
    std::iota(data.begin(), data.end(), 0);

    cudaMemcpy(column->mutable_view().data<int>(),
               data.data(),
               data.size() * sizeof(int),
               cudaMemcpyKind::cudaMemcpyHostToDevice);

    std::vector<std::unique_ptr<cudf::column>> columns;

    columns.emplace_back(std::move(column));

    auto table = std::make_unique<cudf::table>(std::move(columns));

    auto index_info   = cudf::io::column_name_info{""};
    auto column_names = std::vector<cudf::io::column_name_info>({{index_info}});
    auto metadata     = cudf::io::table_metadata{std::move(column_names), {}, {}};

    return cudf::io::table_with_metadata{std::move(table), metadata};
}

TEST_F(TestTritonInferenceStage, SingleRow)
{
    cudf::data_type cudf_data_type{cudf::type_to_id<int>()};

    const std::size_t count = 10;
    const auto dtype        = morpheus::DType::create<int>();

    // Create a 10-number sequence id vector and store them in the tensor.
    auto buffer = std::make_shared<rmm::device_buffer>(count * dtype.item_size(), rmm::cuda_stream_per_thread);
    std::vector<int> seq_ids({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    cudaMemcpy(buffer->data(), seq_ids.data(), count * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);
    auto tensors = morpheus::TensorMap();
    tensors["seq_ids"].swap(morpheus::Tensor::create(buffer, dtype, {count, 1}, {}));

    // create the ControlMessage using the sequence id tensor.
    auto memory  = std::make_shared<morpheus::TensorMemory>(count, std::move(tensors));
    auto table   = create_test_table_with_metadata(count);
    auto meta    = morpheus::MessageMeta::create_from_cpp(std::move(table), 1);
    auto message = std::make_shared<morpheus::ControlMessage>();
    message->payload(meta);
    message->tensors(memory);

    // create the fake triton client used for testing.
    auto triton_client = std::make_unique<ErrorProneTritonClient>();
    auto triton_inference_client =
        std::make_unique<morpheus::TritonInferenceClient>(std::move(triton_client), "", true);
    auto stage = morpheus::InferenceClientStage(std::move(triton_inference_client), "", false, {}, {});

    // manually invoke the stage and iterate through the inference responses
    auto on           = std::make_shared<mrc::coroutines::TestScheduler>();
    auto results_task = [](auto& stage,
                           auto message,
                           auto on) -> mrc::coroutines::Task<std::vector<std::shared_ptr<morpheus::ControlMessage>>> {
        std::vector<std::shared_ptr<morpheus::ControlMessage>> results;

        auto responses_generator = stage.on_data(std::move(message), on);

        auto iter = co_await responses_generator.begin();

        while (iter != responses_generator.end())
        {
            results.emplace_back(std::move(*iter));

            co_await ++iter;
        }

        co_return results;
    }(stage, message, on);

    results_task.resume();

    while (on->resume_next()) {}

    ASSERT_NO_THROW(results_task.promise().result());

    auto results = results_task.promise().result();

    ASSERT_EQ(results.size(), 1);
}

TEST_F(TestTritonInferenceStage, ForceConvert)
{
    using namespace morpheus;
    const TypeId model_type = TypeId::INT32;
    const std::size_t count = 10;

    std::vector<TypeId> test_types = {TypeId::INT8,
                                      TypeId::INT16,
                                      TypeId::INT32,
                                      TypeId::INT64,
                                      TypeId::UINT8,
                                      TypeId::UINT16,
                                      TypeId::UINT32,
                                      TypeId::UINT64};

    for (const auto type_id : test_types)
    {
        for (bool force_convert_inputs : {true, false})
        {
            const bool expect_throw = (type_id != model_type) && !force_convert_inputs;
            const auto dtype        = DType(type_id);

            DVLOG(10) << "Testing type: " << dtype.name() << " with force_convert_inputs: " << force_convert_inputs
                      << " and expect_throw: " << expect_throw;

            // Create a seq_id tensor
            auto md =
                std::make_shared<MemoryDescriptor>(rmm::cuda_stream_per_thread, rmm::mr::get_current_device_resource());
            auto seq_ids_buffer = MatxUtil::create_seq_ids(count, 1, type_id, md);

            auto tensors = TensorMap();
            tensors["seq_ids"].swap(Tensor::create(seq_ids_buffer, dtype, {count, 3}, {}));

            // create the ControlMessage using the sequence id tensor.
            auto memory  = std::make_shared<morpheus::TensorMemory>(count, std::move(tensors));
            auto table   = create_test_table_with_metadata(count);
            auto meta    = morpheus::MessageMeta::create_from_cpp(std::move(table), 1);
            auto message = std::make_shared<morpheus::ControlMessage>();
            message->payload(meta);
            message->tensors(memory);

            // create the fake triton client used for testing.
            auto triton_client = std::make_unique<FakeTritonClient>();
            auto triton_inference_client =
                std::make_unique<morpheus::TritonInferenceClient>(std::move(triton_client), "", force_convert_inputs);
            auto stage = morpheus::InferenceClientStage(std::move(triton_inference_client), "", false, {}, {});

            // manually invoke the stage and iterate through the inference responses
            auto on           = std::make_shared<mrc::coroutines::TestScheduler>();
            auto results_task = [](auto& stage, auto message, auto on)
                -> mrc::coroutines::Task<std::vector<std::shared_ptr<morpheus::ControlMessage>>> {
                std::vector<std::shared_ptr<morpheus::ControlMessage>> results;

                auto responses_generator = stage.on_data(std::move(message), on);

                auto iter = co_await responses_generator.begin();

                while (iter != responses_generator.end())
                {
                    results.emplace_back(std::move(*iter));

                    co_await ++iter;
                }

                co_return results;
            }(stage, message, on);

            results_task.resume();

            while (on->resume_next()) {}

            if (expect_throw)
            {
                ASSERT_THROW(results_task.promise().result(), std::invalid_argument);
            }
            else
            {
                ASSERT_NO_THROW(results_task.promise().result());

                auto results = results_task.promise().result();

                ASSERT_EQ(results.size(), 1);
            }
        }
    }
}
