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

#include "morpheus/stages/triton_inference.hpp"

#include "morpheus/objects/dtype.hpp"          // for DType
#include "morpheus/objects/tensor.hpp"         // for Tensor::create
#include "morpheus/objects/tensor_object.hpp"  // for TensorObject
#include "morpheus/objects/triton_in_out.hpp"  // for TritonInOut
#include "morpheus/types.hpp"                  // for TensorIndex, TensorMap
#include "morpheus/utilities/string_util.hpp"  // for MORPHEUS_CONCAT_STR
#include "morpheus/utilities/tensor_util.hpp"  // for get_elem_count

#include <cuda_runtime.h>  // for cudaMemcpy, cudaMemcpy2D, cudaMemcpyDeviceToHost, cudaMemcpyHostToDevice
#include <glog/logging.h>
#include <http_client.h>
#include <mrc/cuda/common.hpp>  // for MRC_CHECK_CUDA
#include <nlohmann/json.hpp>
#include <rmm/cuda_stream_view.hpp>  // for cuda_stream_per_thread
#include <rmm/device_buffer.hpp>     // for device_buffer

#include <algorithm>  // for min
#include <coroutine>
#include <cstddef>
#include <functional>  // for function
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>  // for runtime_error, out_of_range
#include <string>
#include <utility>
#include <vector>
// IWYU pragma: no_include <initializer_list>

/**
 * @addtogroup stages
 * @{
 * @file
 */

namespace {

/**
 * @brief Checks the status object returned by a Triton client call logging any potential errors.
 *
 */
#define CHECK_TRITON(method) ::InferenceClientStage__check_triton_errors(method, #method, __FILE__, __LINE__);

// Component-private free functions.
void InferenceClientStage__check_triton_errors(triton::client::Error status,
                                               const std::string& methodName,
                                               const std::string& filename,
                                               const int& lineNumber)
{
    if (!status.IsOk())
    {
        std::string err_msg = MORPHEUS_CONCAT_STR("Triton Error while executing '"
                                                  << methodName << "'. Error: " + status.Message() << "\n"
                                                  << filename << "(" << lineNumber << ")");
        LOG(ERROR) << err_msg;
        throw std::runtime_error(err_msg);
    }
}

using namespace morpheus;
using buffer_map_t = std::map<std::string, std::shared_ptr<rmm::device_buffer>>;

static bool is_default_grpc_port(std::string& server_url)
{
    // Check if we are the default gRPC port of 8001 and try 8000 for http client instead
    size_t colon_loc = server_url.find_last_of(':');

    if (colon_loc == -1)
    {
        return false;
    }

    // Check if the port matches 8001
    if (server_url.size() < colon_loc + 1 || server_url.substr(colon_loc + 1) != "8001")
    {
        return false;
    }

    // It matches, change to 8000
    server_url = server_url.substr(0, colon_loc) + ":8000";

    return true;
}

struct TritonInferOperation
{
    bool await_ready() const noexcept
    {
        return false;
    }

    void await_suspend(std::coroutine_handle<> handle)
    {
        CHECK_TRITON(m_client.async_infer(
            [this, handle](triton::client::InferResult* result) {
                m_result.reset(result);
                handle();
            },
            m_options,
            m_inputs,
            m_outputs));
    }

    std::unique_ptr<triton::client::InferResult> await_resume()
    {
        return std::move(m_result);
    }

    ITritonClient& m_client;
    triton::client::InferOptions const& m_options;
    std::vector<TritonInferInput> const& m_inputs;
    std::vector<TritonInferRequestedOutput> const& m_outputs;
    std::unique_ptr<triton::client::InferResult> m_result;
};

}  // namespace

namespace morpheus {

HttpTritonClient::HttpTritonClient(std::string server_url) : m_server_url(std::move(server_url))
{
    // Force the creation of the client
    this->get_client();
}

triton::client::Error HttpTritonClient::is_server_live(bool* live)
{
    return this->get_client().IsServerLive(live);
}

triton::client::Error HttpTritonClient::is_server_ready(bool* ready)
{
    return this->get_client().IsServerReady(ready);
}

triton::client::Error HttpTritonClient::is_model_ready(bool* ready, std::string& model_name)
{
    return this->get_client().IsModelReady(ready, model_name);
}

triton::client::Error HttpTritonClient::model_config(std::string* model_config, std::string& model_name)
{
    return this->get_client().ModelConfig(model_config, model_name);
}

triton::client::Error HttpTritonClient::model_metadata(std::string* model_metadata, std::string& model_name)
{
    return this->get_client().ModelMetadata(model_metadata, model_name);
}

triton::client::Error HttpTritonClient::async_infer(triton::client::InferenceServerHttpClient::OnCompleteFn callback,
                                                    const triton::client::InferOptions& options,
                                                    const std::vector<TritonInferInput>& inputs,
                                                    const std::vector<TritonInferRequestedOutput>& outputs)
{
    std::vector<std::unique_ptr<triton::client::InferInput>> inference_inputs;
    std::vector<triton::client::InferInput*> inference_input_ptrs;

    for (auto& input : inputs)
    {
        triton::client::InferInput* inference_input_ptr;
        triton::client::InferInput::Create(&inference_input_ptr, input.name, input.shape, input.type);

        inference_input_ptr->AppendRaw(input.data);

        inference_input_ptrs.emplace_back(inference_input_ptr);
        inference_inputs.emplace_back(inference_input_ptr);
    }

    std::vector<std::unique_ptr<const triton::client::InferRequestedOutput>> inference_outputs;
    std::vector<const triton::client::InferRequestedOutput*> inference_output_ptrs;

    for (auto& output : outputs)
    {
        triton::client::InferRequestedOutput* inference_output_ptr;
        triton::client::InferRequestedOutput::Create(&inference_output_ptr, output.name);
        inference_output_ptrs.emplace_back(inference_output_ptr);
        inference_outputs.emplace_back(inference_output_ptr);
    }

    triton::client::InferResult* result;

    auto status = this->get_client().Infer(&result, options, inference_input_ptrs, inference_output_ptrs);

    callback(result);

    return status;

    // TODO(cwharris): either fix tests or make this ENV-flagged, as AsyncInfer gives different results.

    //     return this->get_client().AsyncInfer(
    //         [callback](triton::client::InferResult* result) {
    //             callback(result);
    //         },
    //         options,
    //         inference_input_ptrs,
    //         inference_output_ptrs);
}

TritonInferenceClientSession::TritonInferenceClientSession(std::shared_ptr<ITritonClient> client,
                                                           std::string model_name,
                                                           bool force_convert_inputs) :
  m_client(std::move(client)),
  m_model_name(std::move(model_name)),
  m_force_convert_inputs(force_convert_inputs)
{
    // Now load the input/outputs for the model

    bool is_server_live = false;
    CHECK_TRITON(m_client->is_server_live(&is_server_live));
    if (not is_server_live)
    {
        throw std::runtime_error("Server is not live");
    }

    bool is_server_ready = false;
    CHECK_TRITON(m_client->is_server_ready(&is_server_ready));
    if (not is_server_ready)
    {
        throw std::runtime_error("Server is not ready");
    }

    bool is_model_ready = false;
    CHECK_TRITON(m_client->is_model_ready(&is_model_ready, this->m_model_name));
    if (not is_model_ready)
    {
        throw std::runtime_error("Model is not ready");
    }

    std::string model_metadata_json;
    CHECK_TRITON(m_client->model_metadata(&model_metadata_json, this->m_model_name));

    auto model_metadata = nlohmann::json::parse(model_metadata_json);

    std::string model_config_json;
    CHECK_TRITON(m_client->model_config(&model_config_json, this->m_model_name));
    auto model_config = nlohmann::json::parse(model_config_json);

    if (model_config.contains("max_batch_size"))
    {
        m_max_batch_size = model_config.at("max_batch_size").get<TensorIndex>();
    }

    for (auto const& input : model_metadata.at("inputs"))
    {
        auto shape = input.at("shape").get<ShapeType>();

        auto dtype = DType::from_triton(input.at("datatype").get<std::string>());

        size_t bytes = dtype.item_size();

        for (auto& y : shape)
        {
            if (y == -1)
            {
                y = m_max_batch_size;
            }

            bytes *= y;
        }

        m_model_inputs.push_back(TritonInOut{input.at("name").get<std::string>(),
                                             bytes,
                                             DType::from_triton(input.at("datatype").get<std::string>()),
                                             shape,
                                             "",
                                             0});
    }

    for (auto const& output : model_metadata.at("outputs"))
    {
        auto shape = output.at("shape").get<ShapeType>();

        auto dtype = DType::from_triton(output.at("datatype").get<std::string>());

        size_t bytes = dtype.item_size();

        for (auto& y : shape)
        {
            if (y == -1)
            {
                y = m_max_batch_size;
            }

            bytes *= y;
        }

        m_model_outputs.push_back(TritonInOut{output.at("name").get<std::string>(), bytes, dtype, shape, "", 0});
    }
}

std::vector<TensorModelMapping> TritonInferenceClientSession::get_input_mappings(
    std::vector<TensorModelMapping> input_map_overrides)
{
    auto mappings = std::vector<TensorModelMapping>();

    for (auto map : m_model_inputs)
    {
        mappings.emplace_back(TensorModelMapping(map.name, map.name));
    }

    for (auto override : input_map_overrides)
    {
        mappings.emplace_back(override);
    }

    return mappings;
};

std::vector<TensorModelMapping> TritonInferenceClientSession::get_output_mappings(
    std::vector<TensorModelMapping> output_map_overrides)
{
    auto mappings = std::vector<TensorModelMapping>();

    for (auto map : m_model_outputs)
    {
        mappings.emplace_back(TensorModelMapping(map.name, map.name));
    }

    for (auto override : output_map_overrides)
    {
        auto pos = std::find_if(mappings.begin(), mappings.end(), [override](TensorModelMapping m) {
            return m.model_field_name == override.model_field_name;
        });

        if (pos != mappings.end())
        {
            mappings.erase(pos);
        }

        mappings.emplace_back(override);
    }

    return mappings;
}

mrc::coroutines::Task<TensorMap> TritonInferenceClientSession::infer(TensorMap&& inputs)
{
    CHECK_EQ(inputs.size(), m_model_inputs.size()) << "Input tensor count does not match model input count";

    auto element_count = inputs.begin()->second.shape(0);

    for (auto& input : inputs)
    {
        CHECK_EQ(element_count, input.second.shape(0)) << "Input tensors are different sizes";
    }

    TensorMap model_output_tensors;

    // create full inference output
    for (auto& model_output : m_model_outputs)
    {
        ShapeType full_output_shape    = model_output.shape;
        full_output_shape[0]           = element_count;
        auto full_output_element_count = TensorUtils::get_elem_count(full_output_shape);

        auto full_output_buffer = std::make_shared<rmm::device_buffer>(
            full_output_element_count * model_output.datatype.item_size(), rmm::cuda_stream_per_thread);

        ShapeType stride{full_output_shape[1], 1};

        model_output_tensors[model_output.name].swap(
            Tensor::create(std::move(full_output_buffer), model_output.datatype, full_output_shape, stride, 0));
    }

    // process all batches

    for (TensorIndex start = 0; start < element_count; start += m_max_batch_size)
    {
        TensorIndex stop = std::min(start + m_max_batch_size, static_cast<TensorIndex>(element_count));

        // create batch inputs

        std::vector<TritonInferInput> inference_inputs;

        for (auto model_input : m_model_inputs)
        {
            auto inference_input_slice = inputs.at(model_input.name).slice({start, 0}, {stop, -1});

            if (inference_input_slice.dtype() != model_input.datatype)
            {
                if (m_force_convert_inputs)
                {
                    inference_input_slice.swap(inference_input_slice.as_type(model_input.datatype));
                }
                else
                {
                    std::string err_msg = MORPHEUS_CONCAT_STR(
                        "Unexpected dtype for Triton input. Cannot automatically convert dtype due to loss of data."
                        "Input Name: '"
                        << model_input.name << ", Expected: " << model_input.datatype.name()
                        << ", Actual dtype:" << inference_input_slice.dtype().name());
                    throw std::invalid_argument(err_msg);
                }
            }

            inference_inputs.emplace_back(
                TritonInferInput{model_input.name,
                                 {inference_input_slice.shape(0), inference_input_slice.shape(1)},
                                 model_input.datatype.triton_str(),
                                 inference_input_slice.get_host_data()});
        }

        // create batch outputs

        std::vector<TritonInferRequestedOutput> outputs;

        for (auto model_output : m_model_outputs)
        {
            outputs.emplace_back(TritonInferRequestedOutput{model_output.name});
        }

        // infer batch results

        auto options = triton::client::InferOptions(m_model_name);

        auto results = co_await TritonInferOperation(*m_client, options, inference_inputs, outputs);

        // verify batch results and copy to full output tensors

        for (auto model_output : m_model_outputs)
        {
            auto output_tensor = model_output_tensors[model_output.name].slice({start, 0}, {stop, -1});

            std::vector<int64_t> output_shape;

            CHECK_TRITON(results->Shape(model_output.name, &output_shape));

            // Make sure we have at least 2 dims
            while (output_shape.size() < 2)
            {
                output_shape.push_back(1);
            }

            const uint8_t* output_ptr = nullptr;
            size_t output_ptr_size    = 0;

            CHECK_TRITON(results->RawData(model_output.name, &output_ptr, &output_ptr_size));

            // DCHECK_EQ(stop - start, output_shape[0]);
            // DCHECK_EQ(output_tensor.bytes(), output_ptr_size);
            // DCHECK_NOTNULL(output_ptr);            // NOLINT
            // DCHECK_NOTNULL(output_tensor.data());  // NOLINT

            MRC_CHECK_CUDA(cudaMemcpy(output_tensor.data(), output_ptr, output_ptr_size, cudaMemcpyHostToDevice));
        }
    }

    co_return model_output_tensors;
};

TritonInferenceClient::TritonInferenceClient(std::unique_ptr<ITritonClient>&& client,
                                             std::string model_name,
                                             bool force_convert_inputs) :
  m_client(std::move(client)),
  m_model_name(std::move(model_name)),
  m_force_convert_inputs(force_convert_inputs)
{}

std::unique_ptr<IInferenceClientSession> TritonInferenceClient::create_session()
{
    return std::make_unique<TritonInferenceClientSession>(m_client, m_model_name, m_force_convert_inputs);
}

triton::client::InferenceServerHttpClient& HttpTritonClient::get_client()
{
    if (m_fiber_local_client.get() == nullptr)
    {
        // Block in case we need to change the server_url
        std::unique_lock lock(m_client_mutex);

        std::unique_ptr<triton::client::InferenceServerHttpClient> client;

        CHECK_TRITON(triton::client::InferenceServerHttpClient::Create(&client, m_server_url, false));

        bool is_server_live;

        auto status = client->IsServerLive(&is_server_live);

        if (not status.IsOk())
        {
            std::string new_server_url = m_server_url;
            // We are using the default gRPC port, try the default HTTP
            if (is_default_grpc_port(new_server_url))
            {
                LOG(WARNING) << "Failed to connect to Triton at '" << m_server_url
                             << "'. Default gRPC port of (8001) was detected but C++ "
                                "InferenceClientStage uses HTTP protocol. Retrying with default HTTP port (8000)";

                CHECK_TRITON(triton::client::InferenceServerHttpClient::Create(&client, new_server_url, false));

                status = client->IsServerLive(&is_server_live);

                // If that worked, update the server URL
                if (status.IsOk() && is_server_live)
                {
                    m_server_url = new_server_url;
                }
            }
            else if (status.Message().find("Unsupported protocol") != std::string::npos)
            {
                throw std::runtime_error(MORPHEUS_CONCAT_STR(
                    "Failed to connect to Triton at '"
                    << m_server_url
                    << "'. Received 'Unsupported Protocol' error. Are you using the right port? The C++ "
                       "InferenceClientStage uses Triton's HTTP protocol instead of gRPC. Ensure you have "
                       "specified the HTTP port (Default 8000)."));
            }

            if (!status.IsOk())
                throw std::runtime_error(MORPHEUS_CONCAT_STR(
                    "Unable to connect to Triton at '"
                    << m_server_url << "'. Check the URL and port and ensure the server is running."));
        }

        if (!is_server_live)
            throw std::runtime_error(MORPHEUS_CONCAT_STR(
                "Unable to connect to Triton at '"
                << m_server_url
                << "'. Server reported as not live. Check the URL and port and ensure the server is running."));

        m_fiber_local_client.reset(client.release());
    }

    return *m_fiber_local_client;
}
}  // namespace morpheus
