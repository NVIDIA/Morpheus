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

#include "rabbitmq_source.hpp"

#include <SimpleAmqpClient/BasicMessage.h>
#include <SimpleAmqpClient/Channel.h>
#include <SimpleAmqpClient/Envelope.h>
#include <cudf/io/json.hpp>
#include <glog/logging.h>
#include <morpheus/messages/meta.hpp>
#include <pybind11/attr.h>
#include <pybind11/chrono.h>  // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pymrc/utils.hpp>

#include <exception>
#include <sstream>
#include <thread>  // for std::this_thread::sleep_for
#include <utility>

// IWYU pragma: no_include <boost/smart_ptr/detail/operator_bool.hpp>
// IWYU pragma: no_include <boost/smart_ptr/shared_ptr.hpp>

namespace morpheus_rabbit {

RabbitMQSourceStage::RabbitMQSourceStage(const std::string& host,
                                         const std::string& exchange,
                                         const std::string& exchange_type,
                                         const std::string& queue_name,
                                         std::chrono::milliseconds poll_interval) :
  PythonSource(build()),
  m_poll_interval{poll_interval},
  m_channel{AmqpClient::Channel::Create(host)}
{
    m_channel->DeclareExchange(exchange, exchange_type);
    m_queue_name = m_channel->DeclareQueue(queue_name);
    m_channel->BindQueue(m_queue_name, exchange);
}

RabbitMQSourceStage::subscriber_fn_t RabbitMQSourceStage::build()
{
    return [this](rxcpp::subscriber<source_type_t> subscriber) -> void {
        try
        {
            this->source_generator(subscriber);
        } catch (const std::exception& e)
        {
            LOG(ERROR) << "Encountered error while polling RabbitMQ: " << e.what() << std::endl;
            subscriber.on_error(std::make_exception_ptr(e));
            close();
            return;
        }

        close();
        subscriber.on_completed();
    };
}

void RabbitMQSourceStage::source_generator(rxcpp::subscriber<RabbitMQSourceStage::source_type_t> subscriber)
{
    const std::string consumer_tag = m_channel->BasicConsume(m_queue_name, "", true, false);
    while (subscriber.is_subscribed())
    {
        AmqpClient::Envelope::ptr_t envelope;
        if (m_channel->BasicConsumeMessage(consumer_tag, envelope, 0))
        {
            try
            {
                auto table   = from_json(envelope->Message()->Body());
                auto message = MessageMeta::create_from_cpp(std::move(table), 0);
                subscriber.on_next(std::move(message));
            } catch (const std::exception& e)
            {
                LOG(ERROR) << "Error occurred converting RabbitMQ message to Dataframe: " << e.what();
            }
            m_channel->BasicAck(envelope);
        }
        else
        {
            // Sleep when there are no messages
            std::this_thread::sleep_for(m_poll_interval);
        }
    }
}

cudf::io::table_with_metadata RabbitMQSourceStage::from_json(const std::string& body) const
{
    cudf::io::source_info source{body.c_str(), body.size()};
    auto options = cudf::io::json_reader_options::builder(source).lines(true);
    return cudf::io::read_json(options.build());
}

void RabbitMQSourceStage::close()
{
    // disconnect
    if (m_channel)
    {
        m_channel.reset();
    }
}

std::shared_ptr<mrc::segment::Object<RabbitMQSourceStage>> RabbitMQSourceStageInterfaceProxy::init(
    mrc::segment::Builder& builder,
    const std::string& name,
    const std::string& host,
    const std::string& exchange,
    const std::string& exchange_type,
    const std::string& queue_name,
    std::chrono::milliseconds poll_interval)
{
    return builder.construct_object<RabbitMQSourceStage>(
        name, host, exchange, exchange_type, queue_name, poll_interval);
}

namespace py = pybind11;

// Define the pybind11 module m.
PYBIND11_MODULE(rabbitmq_cpp_stage, m)
{
    mrc::pymrc::import(m, "morpheus._lib.messages");

    py::class_<mrc::segment::Object<RabbitMQSourceStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<RabbitMQSourceStage>>>(
        m, "RabbitMQSourceStage", py::multiple_inheritance())
        .def(py::init<>(&RabbitMQSourceStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("host"),
             py::arg("exchange"),
             py::arg("exchange_type") = "fanout",
             py::arg("queue_name")    = "",
             py::arg("poll_interval") = 100ms);
}

}  // namespace morpheus_rabbit
