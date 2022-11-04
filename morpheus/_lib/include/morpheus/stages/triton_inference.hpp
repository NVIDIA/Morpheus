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

#pragma once

#include "morpheus/messages/multi_inference.hpp"
#include "morpheus/messages/multi_response_probs.hpp"
#include "morpheus/objects/triton_in_out.hpp"

#include <http_client.h>
#include <pysrf/node.hpp>
#include <rxcpp/rx.hpp>  // for apply, make_subscriber, observable_member, is_on_error<>::not_void, is_on_next_of<>::not_void, from
#include <srf/channel/status.hpp>          // for Status
#include <srf/node/sink_properties.hpp>    // for SinkProperties<>::sink_type_t
#include <srf/node/source_properties.hpp>  // for SourceProperties<>::source_type_t
#include <srf/segment/builder.hpp>
#include <srf/segment/object.hpp>  // for Object

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** InferenceClientStage********************************/
/**
 * TODO(Documentation)
 */
#pragma GCC visibility push(default)
class InferenceClientStage
  : public srf::pysrf::PythonNode<std::shared_ptr<MultiInferenceMessage>, std::shared_ptr<MultiResponseProbsMessage>>
{
  public:
    using base_t =
        srf::pysrf::PythonNode<std::shared_ptr<MultiInferenceMessage>, std::shared_ptr<MultiResponseProbsMessage>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    InferenceClientStage(std::string model_name,
                         std::string server_url,
                         bool force_convert_inputs,
                         bool use_shared_memory,
                         bool needs_logits,
                         std::map<std::string, std::string> inout_mapping = {});

  private:
    /**
     * TODO(Documentation)
     */
    bool is_default_grpc_port(std::string &server_url);

    /**
     * TODO(Documentation)
     */
    void connect_with_server();

    /**
     * TODO(Documentation)
     */
    subscribe_fn_t build_operator();

    std::string m_model_name;
    std::string m_server_url;
    bool m_force_convert_inputs;
    bool m_use_shared_memory;
    bool m_needs_logits{true};
    std::map<std::string, std::string> m_inout_mapping;

    // Below are settings created during handshake with server
    // std::shared_ptr<triton::client::InferenceServerHttpClient> m_client;
    std::vector<TritonInOut> m_model_inputs;
    std::vector<TritonInOut> m_model_outputs;
    triton::client::InferOptions m_options;
    int m_max_batch_size{-1};

    static std::atomic<size_t> m_request_counter;
};

/****** InferenceClientStageInferenceProxy******************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct InferenceClientStageInterfaceProxy
{
    /**
     * @brief Create and initialize a InferenceClientStage, and return the result.
     */
    static std::shared_ptr<srf::segment::Object<InferenceClientStage>> init(
        srf::segment::Builder &builder,
        const std::string &name,
        std::string model_name,
        std::string server_url,
        bool force_convert_inputs,
        bool use_shared_memory,
        bool needs_logits,
        std::map<std::string, std::string> inout_mapping);
};
#pragma GCC visibility pop
}  // namespace morpheus
