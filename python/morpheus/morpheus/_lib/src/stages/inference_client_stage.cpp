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

#include "morpheus/stages/inference_client_stage.hpp"

#include "morpheus/messages/control.hpp"               // for ControlMessage
#include "morpheus/messages/memory/tensor_memory.hpp"  // for TensorMemory
#include "morpheus/messages/meta.hpp"                  // for MessageMeta
#include "morpheus/objects/data_table.hpp"             // for morpheus
#include "morpheus/objects/dev_mem_info.hpp"           // for DevMemInfo
#include "morpheus/objects/dtype.hpp"                  // for DType
#include "morpheus/objects/tensor.hpp"                 // for Tensor
#include "morpheus/objects/tensor_object.hpp"          // for TensorObject
#include "morpheus/stages/triton_inference.hpp"        // for HttpTritonClient, TritonInferenceClient
#include "morpheus/utilities/matx_util.hpp"            // for MatxUtil

#include <boost/fiber/policy.hpp>  // for launch
#include <cuda_runtime.h>          // for cudaMemcpy2D, cudaMemcpyKind
#include <glog/logging.h>          // for COMPACT_GOOGLE_LOG_WARNING, LOG, LogMessage
#include <mrc/cuda/common.hpp>     // for MRC_CHECK_CUDA
#include <pybind11/pybind11.h>     // for object_api::operator()
#include <rxcpp/rx.hpp>            // for decay_t, trace_activity

#include <chrono>     // for milliseconds, operator""ms, operator<=>, chrono_lite...
#include <compare>    // for operator>, strong_ordering
#include <coroutine>  // for coroutine_handle, suspend_always
#include <memory>     // for shared_ptr, __shared_ptr_access, dynamic_pointer_cast
#include <mutex>      // for unique_lock
#include <ostream>    // for operator<<, basic_ostream
#include <ratio>      // for ratio
#include <stdexcept>  // for runtime_error, invalid_argument
#include <utility>    // for move, pair

namespace {

using namespace morpheus;

static ShapeType get_seq_ids(const std::shared_ptr<ControlMessage>& message)
{
    // Take a copy of the sequence Ids allowing us to map rows in the response to
    // rows in the dataframe The output tensors we store in `reponse_memory` will
    // all be of the same length as the the dataframe. seq_ids has three columns,
    // but we are only interested in the first column.
    auto seq_ids         = message->tensors()->get_tensor("seq_ids");
    const auto item_size = seq_ids.dtype().item_size();

    ShapeType host_seq_ids(message->tensor_count());
    MRC_CHECK_CUDA(cudaMemcpy2D(host_seq_ids.data(),
                                item_size,
                                seq_ids.data(),
                                seq_ids.stride(0) * item_size,
                                item_size,
                                host_seq_ids.size(),
                                cudaMemcpyDeviceToHost));

    return host_seq_ids;
}

static bool has_tensor(std::shared_ptr<ControlMessage> message, std::string const& tensor_name)
{
    return message->tensors()->has_tensor(tensor_name);
}

static TensorObject get_tensor(std::shared_ptr<ControlMessage> message, std::string const& tensor_name)
{
    return message->tensors()->get_tensor(tensor_name);
}

static void reduce_outputs(std::shared_ptr<ControlMessage> const& message, TensorMap& output_tensors)
{
    if (message->payload()->count() == message->tensor_count())
    {
        return;
    }

    // When our tensor lengths are longer than our dataframe we will need to use
    // the seq_ids array to lookup how the values should map back into the
    // dataframe.
    auto host_seq_ids = get_seq_ids(message);

    for (auto& mapping : output_tensors)
    {
        auto& output_tensor = mapping.second;

        ShapeType shape  = output_tensor.get_shape();
        ShapeType stride = output_tensor.get_stride();

        ShapeType reduced_shape{shape};
        reduced_shape[0] = message->payload()->count();

        auto reduced_buffer = MatxUtil::reduce_max(
            DevMemInfo{output_tensor.data(), output_tensor.dtype(), output_tensor.get_memory(), shape, stride},
            host_seq_ids,
            0,
            reduced_shape);

        output_tensor.swap(Tensor::create(std::move(reduced_buffer), output_tensor.dtype(), reduced_shape, stride, 0));
    }
}

static void apply_logits(TensorMap& output_tensors)
{
    for (auto& mapping : output_tensors)
    {
        auto& output_tensor = mapping.second;

        auto shape  = output_tensor.get_shape();
        auto stride = output_tensor.get_stride();

        auto output_buffer = morpheus::MatxUtil::logits(morpheus::DevMemInfo{
            output_tensor.data(), output_tensor.dtype(), output_tensor.get_memory(), shape, stride});

        // For logits the input and output shapes will be the same
        output_tensor.swap(morpheus::Tensor::create(std::move(output_buffer), output_tensor.dtype(), shape, stride, 0));
    }
}

}  // namespace

namespace morpheus {

InferenceClientStage::InferenceClientStage(std::unique_ptr<IInferenceClient>&& client,
                                           std::string model_name,
                                           bool needs_logits,
                                           std::vector<TensorModelMapping> input_mapping,
                                           std::vector<TensorModelMapping> output_mapping) :
  m_model_name(std::move(model_name)),
  m_client(std::move(client)),
  m_needs_logits(needs_logits),
  m_input_mapping(std::move(input_mapping)),
  m_output_mapping(std::move(output_mapping))
{}

struct ExponentialBackoff
{
    std::shared_ptr<mrc::coroutines::Scheduler> m_on;
    std::chrono::milliseconds m_delay;
    std::chrono::milliseconds m_delay_max;

    ExponentialBackoff(std::shared_ptr<mrc::coroutines::Scheduler> on,
                       std::chrono::milliseconds delay_initial,
                       std::chrono::milliseconds delay_max) :
      m_on(std::move(on)),
      m_delay(delay_initial),
      m_delay_max(delay_max)
    {}

    mrc::coroutines::Task<> yield()
    {
        if (m_delay > m_delay_max)
        {
            m_delay = m_delay_max;
        }

        co_await m_on->yield_for(m_delay);

        m_delay *= 2;
    }
};

static std::shared_ptr<ControlMessage> make_response(std::shared_ptr<ControlMessage> message,
                                                     TensorMap&& output_tensor_map)
{
    message->tensors(std::make_shared<TensorMemory>(message->payload()->count(), std::move(output_tensor_map)));
    return message;
}

mrc::coroutines::AsyncGenerator<std::shared_ptr<ControlMessage>> InferenceClientStage::on_data(
    std::shared_ptr<ControlMessage>&& message, std::shared_ptr<mrc::coroutines::Scheduler> on)
{
    int32_t retry_count = 0;

    using namespace std::chrono_literals;

    auto backoff = ExponentialBackoff(on, 100ms, 4000ms);

    while (true)
    {
        auto message_session = m_session;

        try
        {
            // Using the `count` which is the number of rows in the inference tensors.
            // We will check later if this doesn't match the number of rows in the
            // dataframe (`mess_count`). This happens when the size of the input is
            // too large and needs to be broken up in chunks in the pre-process stage.
            // When this is the case we will reduce the rows in the response outputs
            // such that we have a single response for each row int he dataframe.
            // TensorMap output_tensors;
            // buffer_map_t output_buffers;

            if (message_session == nullptr)
            {
                auto lock = std::unique_lock(m_session_mutex);

                if (m_session == nullptr)
                {
                    m_session = m_client->create_session();
                }

                message_session = m_session;
            }

            // We want to prevent entering this section of code if the session is
            // being reset, but we also want this section of code to be entered
            // simultanously by multiple coroutines. To accomplish this, we use a
            // shared lock instead of a unique lock.

            TensorMap model_input_tensors;

            for (auto mapping : message_session->get_input_mappings(m_input_mapping))
            {
                if (has_tensor(message, mapping.tensor_field_name))
                {
                    model_input_tensors[mapping.model_field_name].swap(get_tensor(message, mapping.tensor_field_name));
                }
            }

            auto model_output_tensors = co_await message_session->infer(std::move(model_input_tensors));

            co_await on->yield();

            reduce_outputs(message, model_output_tensors);

            // If we need to do logits, do that here
            if (m_needs_logits)
            {
                apply_logits(model_output_tensors);
            }

            TensorMap output_tensor_map;

            for (auto mapping : message_session->get_output_mappings(m_output_mapping))
            {
                auto pos = model_output_tensors.find(mapping.model_field_name);

                if (pos != model_output_tensors.end())
                {
                    output_tensor_map[mapping.tensor_field_name].swap(
                        std::move(model_output_tensors[mapping.model_field_name]));

                    model_output_tensors.erase(pos);
                }
            }

            auto result = make_response(message, std::move(output_tensor_map));

            co_yield result;

            co_return;

        } catch (std::invalid_argument ex)
        {
            // invalid_argument is terminal, don't attempt to retry
            throw;
        } catch (std::runtime_error ex)
        {
            auto lock = std::unique_lock(m_session_mutex);

            if (m_session == message_session)
            {
                m_session.reset();
            }

            if (m_retry_max >= 0 and ++retry_count > m_retry_max)
            {
                throw;
            }

            LOG(WARNING) << "Exception while processing message for "
                            "InferenceClientStage, attempting retry. ex.what(): "
                         << ex.what();
        } catch (...)
        {
            auto lock = std::unique_lock(m_session_mutex);

            if (m_session == message_session)
            {
                m_session.reset();
            }

            if (m_retry_max >= 0 and ++retry_count > m_retry_max)
            {
                throw;
            }

            LOG(WARNING) << "Exception while processing message for "
                            "InferenceClientStage, attempting retry.";
        }

        co_await backoff.yield();
    }
}

// ************ InferenceClientStageInterfaceProxy********* //
std::shared_ptr<mrc::segment::Object<InferenceClientStage>> InferenceClientStageInterfaceProxy::init(
    mrc::segment::Builder& builder,
    const std::string& name,
    std::string server_url,
    std::string model_name,
    bool needs_logits,
    bool force_convert_inputs,
    std::map<std::string, std::string> input_mappings,
    std::map<std::string, std::string> output_mappings)
{
    std::vector<TensorModelMapping> input_mappings_{};
    std::vector<TensorModelMapping> output_mappings_{};

    for (auto& mapping : input_mappings)
    {
        input_mappings_.emplace_back(TensorModelMapping{mapping.first, mapping.second});
    }

    for (auto& mapping : output_mappings)
    {
        output_mappings_.emplace_back(TensorModelMapping{mapping.first, mapping.second});
    }

    auto triton_client = std::make_unique<HttpTritonClient>(server_url);
    auto triton_inference_client =
        std::make_unique<TritonInferenceClient>(std::move(triton_client), model_name, force_convert_inputs);
    auto stage = builder.construct_object<InferenceClientStage>(
        name, std::move(triton_inference_client), model_name, needs_logits, input_mappings_, output_mappings_);

    return stage;
}

}  // namespace morpheus
