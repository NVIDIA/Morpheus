/**
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

#include <SimpleAmqpClient/Channel.h>
#include <cudf/io/types.hpp>           // for cudf::io::table_with_metadata
#include <morpheus/export.h>           // for exporting symbols
#include <morpheus/messages/meta.hpp>  // for MessageMeta
#include <mrc/segment/builder.hpp>     // for Segment Builder
#include <mrc/segment/object.hpp>      // for Segment Object
#include <pymrc/node.hpp>              // for mrc::pymrc::PythonSource
#include <rxcpp/rx.hpp>

#include <chrono>  // for chrono::milliseconds
#include <memory>  // for shared_ptr
#include <string>
#include <string_view>
#include <thread>

// IWYU pragma: no_include "morpheus/objects/data_table.hpp"
// IWYU pragma: no_include <boost/fiber/context.hpp>

namespace morpheus_rabbit {

using namespace std::literals;
using namespace morpheus;

// pybind11 sets visibility to hidden by default; we want to export our symbols
class MORPHEUS_EXPORT RabbitMQSourceStage : public mrc::pymrc::PythonSource<std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = mrc::pymrc::PythonSource<std::shared_ptr<MessageMeta>>;
    using typename base_t::source_type_t;
    using typename base_t::subscriber_fn_t;

    RabbitMQSourceStage(const std::string& host,
                        const std::string& exchange,
                        const std::string& exchange_type        = "fanout"s,
                        const std::string& queue_name           = ""s,
                        std::chrono::milliseconds poll_interval = 100ms);

    ~RabbitMQSourceStage() override = default;

  private:
    subscriber_fn_t build();
    void source_generator(rxcpp::subscriber<source_type_t> subscriber);
    cudf::io::table_with_metadata from_json(const std::string& body) const;
    void close();

    std::chrono::milliseconds m_poll_interval;
    std::string m_queue_name;
    AmqpClient::Channel::ptr_t m_channel;
};

/****** RabbitMQSourceStageInferenceProxy**********************/
/**
 * @brief Interface proxy, used to insulate Python bindings.
 */
struct MORPHEUS_EXPORT RabbitMQSourceStageInterfaceProxy
{
    /**
     * @brief Create and initialize a RabbitMQSourceStage, and return the result.
     */
    static std::shared_ptr<mrc::segment::Object<RabbitMQSourceStage>> init(mrc::segment::Builder& builder,
                                                                           const std::string& name,
                                                                           const std::string& host,
                                                                           const std::string& exchange,
                                                                           const std::string& exchange_type,
                                                                           const std::string& queue_name,
                                                                           std::chrono::milliseconds poll_interval);
};

}  // namespace morpheus_rabbit
