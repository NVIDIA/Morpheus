/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/messages/memory/inference_memory.hpp"       // for InferenceMemory
#include "morpheus/messages/memory/response_memory_probs.hpp"  // for ResponseMemoryProbs
#include "morpheus/messages/memory/tensor_memory.hpp"          // for TensorMemory::tensor_map_t
#include "morpheus/messages/multi_response_probs.hpp"
#include "morpheus/objects/dev_mem_info.hpp"  // for DevMemInfo
#include "morpheus/objects/tensor.hpp"
#include "morpheus/objects/tensor_object.hpp"  // for TensorIndex, TensorObject
#include "morpheus/objects/triton_in_out.hpp"
#include "morpheus/utilities/matx_util.hpp"
#include "morpheus/utilities/stage_util.hpp"
#include "morpheus/utilities/string_util.hpp"  // for MORPHEUS_CONCAT_STR
#include "morpheus/utilities/type_util.hpp"
#include "morpheus/utilities/type_util_detail.hpp"  // for DataType

#include <cuda_runtime.h>  // for cudaMemcpy, cudaMemcpy2D, cudaMemcpyDeviceToHost, cudaMemcpyHostToDevice
#include <glog/logging.h>
#include <http_client.h>
#include <nlohmann/json.hpp>
#include <pysrf/node.hpp>
#include <rmm/cuda_stream_view.hpp>  // for cuda_stream_per_thread
#include <rmm/device_buffer.hpp>     // for device_buffer
#include <srf/cuda/common.hpp>       // for SRF_CHECK_CUDA

#include <algorithm>  // for min
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>  // for multiplies
#include <memory>
#include <numeric>  // for accumulate
#include <sstream>
#include <stdexcept>    // for runtime_error, out_of_range
#include <type_traits>  // for declval
#include <utility>
// IWYU pragma: no_include <initializer_list>

#define CHECK_TRITON(method) ::InferenceClientStage__check_triton_errors(method, #method, __FILE__, __LINE__);

namespace {
// Component-private free functions.
void InferenceClientStage__check_triton_errors(triton::client::Error status,
                                               const std::string &methodName,
                                               const std::string &filename,
                                               const int &lineNumber)
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

template <typename IndexT>
inline IndexT get_elem_count(const std::vector<IndexT> &shape)
{
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
}
}  // namespace

namespace morpheus {

std::atomic<size_t> InferenceClientStage::m_request_counter = 0;

// Component public implementations
// ************ InferenceClientStage ************************* //
InferenceClientStage::InferenceClientStage(std::string model_name,
                                           std::string server_url,
                                           bool force_convert_inputs,
                                           bool use_shared_memory,
                                           bool needs_logits,
                                           std::map<std::string, std::string> inout_mapping) :
  PythonNode(base_t::op_factory_from_sub_fn(build_operator())),
  m_model_name(std::move(model_name)),
  m_server_url(std::move(server_url)),
  m_force_convert_inputs(force_convert_inputs),
  m_use_shared_memory(use_shared_memory),
  m_needs_logits(needs_logits),
  m_inout_mapping(std::move(inout_mapping)),
  m_options(m_model_name)
{
    // Connect with the server to setup the inputs/outputs
    this->connect_with_server();  // TODO(Devin)
}

InferenceClientStage::subscribe_fn_t InferenceClientStage::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        std::unique_ptr<triton::client::InferenceServerHttpClient> client;

        CHECK_TRITON(triton::client::InferenceServerHttpClient::Create(&client, m_server_url, false));

        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output, &client](sink_type_t x) {
                // When our tensor lengths are longer than our dataframe we will need to use the seq_ids
                // array to lookup how the values should map back into the dataframe
                const bool needs_seq_ids = x->mess_count != x->count;
                std::map<std::string, TensorObject> response_outputs;

                // Create the output memory blocks
                for (auto &model_output : m_model_outputs)
                {
                    std::vector<TensorIndex> total_shape{model_output.shape.begin(), model_output.shape.end()};

                    // First dimension will always end up being the number of rows in the dataframe
                    total_shape[0]  = static_cast<TensorIndex>(x->mess_count);
                    auto elem_count = get_elem_count(total_shape);

                    // Create the output memory
                    auto output_buffer = std::make_shared<rmm::device_buffer>(
                        elem_count * model_output.datatype.item_size(), rmm::cuda_stream_per_thread);

                    response_outputs[model_output.mapped_name] = Tensor::create(
                        std::move(output_buffer), model_output.datatype, total_shape, std::vector<TensorIndex>{}, 0);
                }

                // This will be the final output of all mini-batches
                auto response_mem_probs =
                    std::make_shared<ResponseMemoryProbs>(x->mess_count, std::move(response_outputs));
                auto response = std::make_shared<MultiResponseProbsMessage>(x->meta,
                                                                            x->mess_offset,
                                                                            x->mess_count,
                                                                            std::move(response_mem_probs),
                                                                            0,
                                                                            response_mem_probs->count);

                std::unique_ptr<std::vector<int32_t>> host_seq_ids{nullptr};
                if (needs_seq_ids)
                {
                    // Take a copy of the sequence Ids allowing us to map rows in the response to rows in the dataframe
                    // The output tensors we store in `reponse_memory` will all be of the same length as the the
                    // dataframe. seq_ids has three columns, but we are only interested in the first column.
                    auto seq_ids         = x->get_input("seq_ids");
                    const auto item_size = seq_ids.dtype().item_size();

                    host_seq_ids = std::make_unique<std::vector<int32_t>>(x->count);
                    SRF_CHECK_CUDA(cudaMemcpy2D(host_seq_ids->data(),
                                                item_size,
                                                seq_ids.data(),
                                                seq_ids.stride(0) * item_size,
                                                item_size,
                                                host_seq_ids->size(),
                                                cudaMemcpyDeviceToHost));
                }

                for (size_t i = 0; i < x->count; i += m_max_batch_size)
                {
                    triton::client::InferInput *input1;

                    size_t start = i;
                    size_t stop  = std::min(i + m_max_batch_size, x->count);

                    sink_type_t mini_batch_input = x->get_slice(start, stop);

                    size_t out_start = start;
                    size_t out_stop  = stop;
                    if (needs_seq_ids)
                    {
                        out_start = (*host_seq_ids)[out_start];
                        if (out_stop < host_seq_ids->size())
                        {
                            out_stop = (*host_seq_ids)[out_stop];
                        }
                        else
                        {
                            out_stop = x->mess_count;
                        }
                    }

                    source_type_t mini_batch_output = response->get_slice(out_start, out_stop);

                    // Iterate on the model inputs in case the model takes less than what tensors are available
                    std::vector<std::pair<std::shared_ptr<triton::client::InferInput>, std::vector<uint8_t>>>
                        saved_inputs = foreach_map(m_model_inputs, [this, &mini_batch_input](auto const &model_input) {
                            DCHECK(mini_batch_input->memory->has_tensor(model_input.mapped_name))
                                << "Model input '" << model_input.mapped_name << "' not found in InferenceMemory";

                            auto const &inp_tensor = mini_batch_input->get_input(model_input.mapped_name);

                            // Convert to the right type. Make shallow if necessary
                            auto final_tensor = inp_tensor.as_type(model_input.datatype);

                            std::vector<uint8_t> inp_data = final_tensor.get_host_data();

                            // Test
                            triton::client::InferInput *inp_ptr;

                            triton::client::InferInput::Create(&inp_ptr,
                                                               model_input.name,
                                                               {inp_tensor.shape(0), inp_tensor.shape(1)},
                                                               model_input.datatype.triton_str());
                            std::shared_ptr<triton::client::InferInput> inp_shared;
                            inp_shared.reset(inp_ptr);

                            inp_ptr->AppendRaw(inp_data);

                            return std::make_pair(inp_shared, std::move(inp_data));
                        });

                    std::vector<std::shared_ptr<const triton::client::InferRequestedOutput>> saved_outputs =
                        foreach_map(m_model_outputs, [this](auto const &model_output) {
                            // Generate the outputs to be requested.
                            triton::client::InferRequestedOutput *out_ptr;

                            triton::client::InferRequestedOutput::Create(&out_ptr, model_output.name);
                            std::shared_ptr<const triton::client::InferRequestedOutput> out_shared;
                            out_shared.reset(out_ptr);

                            return out_shared;
                        });

                    std::vector<triton::client::InferInput *> inputs =
                        foreach_map(saved_inputs, [](auto x) { return x.first.get(); });

                    std::vector<const triton::client::InferRequestedOutput *> outputs =
                        foreach_map(saved_outputs, [](auto x) { return x.get(); });

                    // this->segment().resources().fiber_pool().enqueue([client, output](){});

                    triton::client::InferResult *results;

                    // Make a copy of the options object and set the request ID
                    auto options        = m_options;
                    options.request_id_ = m_request_counter++;

                    CHECK_TRITON(client->Infer(&results, options, inputs, outputs));

                    for (auto &model_output : m_model_outputs)
                    {
                        std::vector<int64_t> output_shape;

                        CHECK_TRITON(results->Shape(model_output.name, &output_shape));

                        // Make sure we have at least 2 dims
                        while (output_shape.size() < 2)
                        {
                            output_shape.push_back(1);
                        }

                        const uint8_t *output_ptr = nullptr;
                        size_t output_ptr_size    = 0;
                        CHECK_TRITON(results->RawData(model_output.name, &output_ptr, &output_ptr_size));

                        auto output_buffer =
                            std::make_shared<rmm::device_buffer>(output_ptr_size, rmm::cuda_stream_per_thread);

                        SRF_CHECK_CUDA(
                            cudaMemcpy(output_buffer->data(), output_ptr, output_ptr_size, cudaMemcpyHostToDevice));

                        if (needs_seq_ids && output_shape[0] != mini_batch_output->count)
                        {
                            // Since we are working with slices of both the input and the output, the seq_ids have
                            // already been applied to the output's start & stop, so we only need to reduce the
                            // response tensort when the size doesn't match our output
                            std::vector<int64_t> mapped_output_shape{output_shape};
                            mapped_output_shape[0] = mini_batch_output->count;

                            size_t element_count = get_elem_count(output_shape);

                            // Triton results are always in row-major as required by the KServe protocol
                            // https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#tensor-data
                            std::vector<int64_t> stride{output_shape[1], 1};
                            output_buffer = MatxUtil::reduce_max(
                                DevMemInfo{element_count, model_output.datatype.type_id(), output_buffer, 0},
                                *host_seq_ids,
                                mini_batch_input->offset,
                                output_shape,
                                stride,
                                mapped_output_shape);
                            output_shape = std::move(mapped_output_shape);
                        }

                        // If we need to do logits, do that here
                        if (m_needs_logits)
                        {
                            size_t element_count = get_elem_count(output_shape);
                            output_buffer        = MatxUtil::logits(
                                DevMemInfo{element_count, model_output.datatype.type_id(), output_buffer, 0});
                        }

                        mini_batch_output->set_output(
                            model_output.mapped_name,
                            Tensor::create(std::move(output_buffer),
                                           model_output.datatype,
                                           std::vector<TensorIndex>{static_cast<TensorIndex>(output_shape[0]),
                                                                    static_cast<TensorIndex>(output_shape[1])},
                                           std::vector<TensorIndex>{},
                                           0));
                    }
                }
                output.on_next(std::move(response));
            },
            [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
            [&]() { output.on_completed(); }));
    };
}

void InferenceClientStage::connect_with_server()
{
    std::string server_url = m_server_url;

    std::unique_ptr<triton::client::InferenceServerHttpClient> client;

    auto result = triton::client::InferenceServerHttpClient::Create(&client, server_url, false);

    // Now load the input/outputs for the model
    bool is_server_live = false;

    triton::client::Error status = client->IsServerLive(&is_server_live);

    if (!status.IsOk())
    {
        if (this->is_default_grpc_port(server_url))
        {
            LOG(WARNING) << "Failed to connect to Triton at '" << m_server_url
                         << "'. Default gRPC port of (8001) was detected but C++ "
                            "InferenceClientStage uses HTTP protocol. Retrying with default HTTP port (8000)";

            // We are using the default gRPC port, try the default HTTP
            std::unique_ptr<triton::client::InferenceServerHttpClient> unique_client;

            auto result = triton::client::InferenceServerHttpClient::Create(&unique_client, server_url, false);

            client = std::move(unique_client);

            status = client->IsServerLive(&is_server_live);
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
            throw std::runtime_error(
                MORPHEUS_CONCAT_STR("Unable to connect to Triton at '"
                                    << m_server_url << "'. Check the URL and port and ensure the server is running."));
    }

    // Save this for new clients
    m_server_url = server_url;

    if (!is_server_live)
        throw std::runtime_error("Server is not live");

    bool is_server_ready = false;
    CHECK_TRITON(client->IsServerReady(&is_server_ready));

    if (!is_server_ready)
        throw std::runtime_error("Server is not ready");

    bool is_model_ready = false;
    CHECK_TRITON(client->IsModelReady(&is_model_ready, this->m_model_name));

    if (!is_model_ready)
        throw std::runtime_error("Model is not ready");

    std::string model_metadata_json;
    CHECK_TRITON(client->ModelMetadata(&model_metadata_json, this->m_model_name));

    auto model_metadata = nlohmann::json::parse(model_metadata_json);

    std::string model_config_json;
    CHECK_TRITON(client->ModelConfig(&model_config_json, this->m_model_name));

    auto model_config = nlohmann::json::parse(model_config_json);

    if (model_config.contains("max_batch_size"))
    {
        m_max_batch_size = model_config.at("max_batch_size").get<int>();
    }

    for (auto const &input : model_metadata.at("inputs"))
    {
        auto shape = input.at("shape").get<std::vector<int>>();

        auto dtype = DType::from_triton(input.at("datatype").get<std::string>());

        size_t bytes = dtype.item_size();

        for (auto &y : shape)
        {
            if (y == -1)
            {
                y = m_max_batch_size;
            }

            bytes *= y;
        }

        std::string mapped_name = input.at("name").get<std::string>();

        if (m_inout_mapping.find(mapped_name) != m_inout_mapping.end())
        {
            mapped_name = m_inout_mapping[mapped_name];
        }

        m_model_inputs.push_back(TritonInOut{input.at("name").get<std::string>(),
                                             bytes,
                                             DType::from_triton(input.at("datatype").get<std::string>()),
                                             shape,
                                             mapped_name,
                                             0});
    }

    for (auto const &output : model_metadata.at("outputs"))
    {
        auto shape = output.at("shape").get<std::vector<int>>();

        auto dtype = DType::from_triton(output.at("datatype").get<std::string>());

        size_t bytes = dtype.item_size();

        for (auto &y : shape)
        {
            if (y == -1)
            {
                y = m_max_batch_size;
            }

            bytes *= y;
        }

        std::string mapped_name = output.at("name").get<std::string>();

        if (m_inout_mapping.find(mapped_name) != m_inout_mapping.end())
        {
            mapped_name = m_inout_mapping[mapped_name];
        }

        m_model_outputs.push_back(
            TritonInOut{output.at("name").get<std::string>(), bytes, dtype, shape, mapped_name, 0});
    }
}

bool InferenceClientStage::is_default_grpc_port(std::string &server_url)
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

// ************ InferenceClientStageInterfaceProxy********* //
std::shared_ptr<srf::segment::Object<InferenceClientStage>> InferenceClientStageInterfaceProxy::init(
    srf::segment::Builder &builder,
    const std::string &name,
    std::string model_name,
    std::string server_url,
    bool force_convert_inputs,
    bool use_shared_memory,
    bool needs_logits,
    std::map<std::string, std::string> inout_mapping)
{
    auto stage = builder.construct_object<InferenceClientStage>(
        name, model_name, server_url, force_convert_inputs, use_shared_memory, needs_logits, inout_mapping);

    return stage;
}
}  // namespace morpheus
