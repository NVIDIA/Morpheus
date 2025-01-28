/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/stages/http_server_source_stage.hpp"

#include <pybind11/pybind11.h>  // for cast
#include <pymrc/utils.hpp>      // for cast_from_pyobject

namespace morpheus {

void make_output_message(std::shared_ptr<MessageMeta>& incoming_message,
                         control_message_task_t* task,
                         morpheus::utilities::json_t&& http_fields,
                         std::shared_ptr<MessageMeta>& out_message)
{
    DCHECK_EQ(task, nullptr) << "Tasks are not supported for MessageMeta";
    out_message.swap(incoming_message);
}

void make_output_message(std::shared_ptr<MessageMeta>& incoming_message,
                         control_message_task_t* task,
                         morpheus::utilities::json_t&& http_fields,
                         std::shared_ptr<ControlMessage>& out_message)
{
    utilities::json_t cm_config = {{"metadata", {{"http_fields", http_fields}}}};
    auto cm_msg                 = std::make_shared<ControlMessage>(cm_config);
    cm_msg->payload(incoming_message);
    if (task)
    {
        cm_msg->add_task(task->first, task->second);
    }
    out_message.swap(cm_msg);
}

// ************ HttpServerSourceStageInterfaceProxy ************ //
std::shared_ptr<mrc::segment::Object<HttpServerSourceStage<MessageMeta>>>
HttpServerSourceStageInterfaceProxy::init_meta(mrc::segment::Builder& builder,
                                               const std::string& name,
                                               std::string bind_address,
                                               unsigned short port,
                                               std::string endpoint,
                                               std::string live_endpoint,
                                               std::string ready_endpoint,
                                               std::string method,
                                               std::string live_method,
                                               std::string ready_method,
                                               unsigned accept_status,
                                               float sleep_time,
                                               long queue_timeout,
                                               std::size_t max_queue_size,
                                               unsigned short num_server_threads,
                                               std::size_t max_payload_size,
                                               int64_t request_timeout,
                                               bool lines,
                                               std::size_t stop_after)
{
    return builder.construct_object<HttpServerSourceStage<MessageMeta>>(name,
                                                                        std::move(bind_address),
                                                                        port,
                                                                        std::move(endpoint),
                                                                        std::move(live_endpoint),
                                                                        std::move(ready_endpoint),
                                                                        std::move(method),
                                                                        std::move(live_method),
                                                                        std::move(ready_method),
                                                                        accept_status,
                                                                        sleep_time,
                                                                        queue_timeout,
                                                                        max_queue_size,
                                                                        num_server_threads,
                                                                        max_payload_size,
                                                                        std::chrono::seconds(request_timeout),
                                                                        lines,
                                                                        stop_after,
                                                                        nullptr);
}

std::shared_ptr<mrc::segment::Object<HttpServerSourceStage<ControlMessage>>>
HttpServerSourceStageInterfaceProxy::init_cm(mrc::segment::Builder& builder,
                                             const std::string& name,
                                             std::string bind_address,
                                             unsigned short port,
                                             std::string endpoint,
                                             std::string live_endpoint,
                                             std::string ready_endpoint,
                                             std::string method,
                                             std::string live_method,
                                             std::string ready_method,
                                             unsigned accept_status,
                                             float sleep_time,
                                             long queue_timeout,
                                             std::size_t max_queue_size,
                                             unsigned short num_server_threads,
                                             std::size_t max_payload_size,
                                             int64_t request_timeout,
                                             bool lines,
                                             std::size_t stop_after,
                                             const pybind11::object& task_type,
                                             const pybind11::object& task_payload)
{
    std::unique_ptr<control_message_task_t> task{nullptr};

    if (!task_type.is_none() && !task_payload.is_none())
    {
        task = std::make_unique<control_message_task_t>(pybind11::cast<std::string>(task_type),
                                                        mrc::pymrc::cast_from_pyobject(task_payload));
    }

    return builder.construct_object<HttpServerSourceStage<ControlMessage>>(name,
                                                                           std::move(bind_address),
                                                                           port,
                                                                           std::move(endpoint),
                                                                           std::move(live_endpoint),
                                                                           std::move(ready_endpoint),
                                                                           std::move(method),
                                                                           std::move(live_method),
                                                                           std::move(ready_method),
                                                                           accept_status,
                                                                           sleep_time,
                                                                           queue_timeout,
                                                                           max_queue_size,
                                                                           num_server_threads,
                                                                           max_payload_size,
                                                                           std::chrono::seconds(request_timeout),
                                                                           lines,
                                                                           stop_after,
                                                                           std::move(task));
}
}  // namespace morpheus
