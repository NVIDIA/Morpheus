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

#pragma once

#include "morpheus/export.h"                   // for exporting symbols
#include "morpheus/messages/control.hpp"       // for ControlMessage
#include "morpheus/messages/meta.hpp"          // for MessageMeta
#include "morpheus/utilities/http_server.hpp"  // for HttpServer
#include "morpheus/utilities/json_types.hpp"   // for control_message_task_t & json_t

#include <boost/beast/http/status.hpp>        // for int_to_status, status
#include <boost/fiber/buffered_channel.hpp>   // for buffered_channel
#include <boost/fiber/channel_op_status.hpp>  // for channel_op_status
#include <boost/fiber/operations.hpp>         // for sleep_for
#include <cudf/io/json.hpp>                   // for json_reader_options & read_json
#include <cudf/io/types.hpp>                  // for table_with_metadata
#include <glog/logging.h>                     // for CHECK & LOG
#include <mrc/segment/builder.hpp>            // for segment::Builder
#include <mrc/segment/object.hpp>             // for segment::Object
#include <nlohmann/json.hpp>                  // for json
#include <pybind11/pytypes.h>                 // for pybind11::object
#include <pymrc/node.hpp>                     // for PythonSource
#include <rxcpp/rx.hpp>                       // for subscriber

#include <atomic>     // for atomic
#include <chrono>     // for duration
#include <cstddef>    // for size_t
#include <cstdint>    // for int64_t
#include <exception>  // for std::exception
#include <memory>     // for shared_ptr & unique_ptr
#include <sstream>    // needed by GLOG
#include <stdexcept>  // for std::runtime_error
#include <string>     // for string & to_string
#include <tuple>      // for make_tuple
#include <utility>    // for std::move & pair
#include <vector>     // for vector
// IWYU thinks we're using thread::operator<<
// IWYU pragma: no_include <thread>
// IWYU thinks we need the http.hpp header for int_to_status, but it's defined in status.hpp
// IWYU pragma: no_include <boost/beast/http.hpp>

namespace morpheus {
using table_with_http_fields_t = std::pair<cudf::io::table_with_metadata, morpheus::utilities::json_t>;
using table_t                  = std::unique_ptr<table_with_http_fields_t>;

using request_queue_t = boost::fibers::buffered_channel<table_t>;

class SourceStageStopAfter : public std::exception
{};

// Per type overloads for producing the output message
void make_output_message(std::shared_ptr<MessageMeta>& incoming_message,
                         control_message_task_t* task,
                         morpheus::utilities::json_t&& http_fields,
                         std::shared_ptr<MessageMeta>& out_message);

void make_output_message(std::shared_ptr<MessageMeta>& incoming_message,
                         control_message_task_t* task,
                         morpheus::utilities::json_t&& http_fields,
                         std::shared_ptr<ControlMessage>& out_message);

/****** HttpServerSourceStage *************************************/

/**
 * @addtogroup stages
 * @{
 * @file
 */
template <typename OutputT>
class MORPHEUS_EXPORT HttpServerSourceStage : public mrc::pymrc::PythonSource<std::shared_ptr<OutputT>>
{
  public:
    using base_t = mrc::pymrc::PythonSource<std::shared_ptr<OutputT>>;
    using typename base_t::source_type_t;
    using typename base_t::subscriber_fn_t;

    /**
     * @brief Constructor for the HttpServerSourceStage
     *
     * @param bind_address The IP address to bind the server to
     * @param port The TCP port to bind the server to
     * @param endpoint The endpoint to listen for messages on
     * @param live_endpoint The endpoint to check if the server is running
     * @param ready_endpoint The endpoint to check if the server is ready to accept messages
     * @param method The HTTP method to accept requests on the `endpoint`
     * @param live_method The HTTP method to accept requests on the `live_endpoint`
     * @param ready_method The HTTP method accept requests on the `ready_endpoint`
     * @param accept_status The HTTP status code to return when a message is accepted
     * @param sleep_time The time to sleep when the queue is empty
     * @param queue_timeout The time to wait for the queue to accept a message
     * @param max_queue_size The maximum number of messages to queue prior to blocking incoming requests
     * @param num_server_threads The number of threads to run the server on
     * @param max_payload_size The maximum size of the payload
     * @param request_timeout The time to wait for a request to complete
     * @param lines If `false`, the HTTP server will expect each request to be a JSON array of objects. If `true`, the
     * HTTP server will expect each request to be a JSON object per line.
     * @param stop_after The number of records to emit before stopping. Useful for testing, disabled if `0`.
     * @param task When `OutputT=ControlMessage`, optional task to be added to all outgoing messagess, triggers an
     * assertion error for all other types.
     */
    HttpServerSourceStage(std::string bind_address                     = "127.0.0.1",
                          unsigned short port                          = 8080,
                          std::string endpoint                         = "/message",
                          std::string live_endpoint                    = "/live",
                          std::string ready_endpoint                   = "/ready",
                          std::string method                           = "POST",
                          std::string live_method                      = "GET",
                          std::string ready_method                     = "GET",
                          unsigned accept_status                       = 201,
                          float sleep_time                             = 0.1f,
                          long queue_timeout                           = 5,
                          std::size_t max_queue_size                   = 1024,
                          unsigned short num_server_threads            = 1,
                          std::size_t max_payload_size                 = DefaultMaxPayloadSize,
                          std::chrono::seconds request_timeout         = std::chrono::seconds(30),
                          bool lines                                   = false,
                          std::size_t stop_after                       = 0,
                          std::unique_ptr<control_message_task_t> task = nullptr);
    ~HttpServerSourceStage() override
    {
        close();
    }

    void close();

  private:
    subscriber_fn_t build();
    void source_generator(rxcpp::subscriber<source_type_t> subscriber);

    std::atomic<int> m_queue_cnt = 0;
    std::chrono::steady_clock::duration m_sleep_time;
    std::chrono::duration<long> m_queue_timeout;
    std::unique_ptr<HttpServer> m_server;
    request_queue_t m_queue;
    std::size_t m_max_queue_size;
    std::size_t m_stop_after;
    std::size_t m_records_emitted{0};
    std::unique_ptr<control_message_task_t> m_task{nullptr};
};

/****** HttpServerSourceStageInterfaceProxy***********************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MORPHEUS_EXPORT HttpServerSourceStageInterfaceProxy
{
    /**
     * @brief Create and initialize a HttpServerSourceStage that emits MessageMeta's, and return the result
     *
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @param bind_address The IP address to bind the server to
     * @param port The TCP port to bind the server to
     * @param endpoint The endpoint to listen for messages on
     * @param live_endpoint The endpoint to check if the server is running
     * @param ready_endpoint The endpoint to check if the server is ready to accept messages
     * @param method The HTTP method to accept requests on the `endpoint`
     * @param live_method The HTTP method to accept requests on the `live_endpoint`
     * @param ready_method The HTTP method accept requests on the `ready_endpoint`
     * @param accept_status The HTTP status code to return when a message is accepted
     * @param sleep_time The time to sleep when the queue is empty
     * @param queue_timeout The time to wait for the queue to accept a message
     * @param max_queue_size The maximum number of messages to queue prior to blocking incoming requests
     * @param num_server_threads The number of threads to run the server on
     * @param max_payload_size The maximum size of the payload
     * @param request_timeout The time to wait for a request to complete
     * @param lines If `False`, the HTTP server will expect each request to be a JSON array of objects. If `True`, the
     * HTTP server will expect each request to be a JSON object per line.
     * @param stop_after The number of records to emit before stopping. Useful for testing, disabled if `0`.
     */
    static std::shared_ptr<mrc::segment::Object<HttpServerSourceStage<MessageMeta>>> init_meta(
        mrc::segment::Builder& builder,
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
        std::size_t stop_after);

    /**
     * @brief Create and initialize a HttpServerSourceStage that emits ControlMessage's, and return the result
     *
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @param bind_address The IP address to bind the server to
     * @param port The TCP port to bind the server to
     * @param endpoint The endpoint to listen for messages on
     * @param live_endpoint The endpoint to check if the server is running
     * @param ready_endpoint The endpoint to check if the server is ready to accept messages
     * @param method The HTTP method to accept requests on the `endpoint`
     * @param live_method The HTTP method to accept requests on the `live_endpoint`
     * @param ready_method The HTTP method accept requests on the `ready_endpoint`
     * @param accept_status The HTTP status code to return when a message is accepted
     * @param sleep_time The time to sleep when the queue is empty
     * @param queue_timeout The time to wait for the queue to accept a message
     * @param max_queue_size The maximum number of messages to queue prior to blocking incoming requests
     * @param num_server_threads The number of threads to run the server on
     * @param max_payload_size The maximum size of the payload
     * @param request_timeout The time to wait for a request to complete
     * @param lines If `False`, the HTTP server will expect each request to be a JSON array of objects. If `True`, the
     * HTTP server will expect each request to be a JSON object per line.
     * @param stop_after The number of records to emit before stopping. Useful for testing, disabled if `0`.
     * @param task_type Optional task type to be added to all outgoing messages. When not `None`, then `task_payload`
     * must also be not `None`, and vice versa.
     * @param task_payload Optional json object describing the task to be added to all outgoing messages. When not
     * `None`, then `task_type` must also be not `None`, and vice versa.
     */
    static std::shared_ptr<mrc::segment::Object<HttpServerSourceStage<ControlMessage>>> init_cm(
        mrc::segment::Builder& builder,
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
        const pybind11::object& task_payload);
};

template <typename OutputT>
HttpServerSourceStage<OutputT>::HttpServerSourceStage(std::string bind_address,
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
                                                      std::chrono::seconds request_timeout,
                                                      bool lines,
                                                      std::size_t stop_after,
                                                      std::unique_ptr<control_message_task_t> task) :
  base_t(build()),
  m_max_queue_size{max_queue_size},
  m_sleep_time{std::chrono::milliseconds(static_cast<long int>(sleep_time))},
  m_queue_timeout{queue_timeout},
  m_queue{max_queue_size},
  m_stop_after{stop_after},
  m_task{std::move(task)}
{
    CHECK(boost::beast::http::int_to_status(accept_status) != boost::beast::http::status::unknown)
        << "Invalid HTTP status code: " << accept_status;

    request_handler_fn_t parser = [this, accept_status, lines](const tcp_endpoint_t& tcp_endpoint,
                                                               const request_t& request) {
        // This function is called from one of the HTTPServer's worker threads, avoid performing any additional work
        // here beyond what is strictly nessary to return a valid response to the client. We parse the payload here,
        // that way we can return an appropriate error message if the payload is invalid however we stop avoid
        // constructing a MessageMeta object since that would require grabbing the Python GIL, instead we push the
        // libcudf table to the queue and let the subscriber handle the conversion to MessageMeta.
        table_t table{nullptr};

        try
        {
            std::string body{request.body()};
            cudf::io::source_info source{body.c_str(), body.size()};
            auto options    = cudf::io::json_reader_options::builder(source).lines(lines);
            auto cudf_table = cudf::io::read_json(options.build());

            auto http_fields = request_headers_to_json(tcp_endpoint, request);
            table = std::make_unique<table_with_http_fields_t>(std::move(cudf_table), std::move(http_fields));
        } catch (const std::exception& e)
        {
            // We want to log the exception locally, but we don't want to include the exception message in the response
            // since that may leak sensitive information
            std::string error_msg = "Error occurred converting HTTP payload to Dataframe";
            LOG(ERROR) << error_msg << ": " << e.what();
            return std::make_tuple(400u, "text/plain", error_msg, nullptr);
        }

        try
        {
            // NOLINTNEXTLINE(clang-diagnostic-unused-value)
            DCHECK_NOTNULL(table);
            auto queue_status = m_queue.push_wait_for(std::move(table), m_queue_timeout);

            if (queue_status == boost::fibers::channel_op_status::success)
            {
                ++m_queue_cnt;
                return std::make_tuple(accept_status, "text/plain", std::string(), nullptr);
            }

            std::string error_msg = "HTTP payload queue is ";
            switch (queue_status)
            {
            case boost::fibers::channel_op_status::full:
            case boost::fibers::channel_op_status::timeout: {
                error_msg += "full";
                break;
            }

            case boost::fibers::channel_op_status::closed: {
                error_msg += "closed";
                break;
            }
            default: {
                error_msg += "in an unknown state";
                break;
            }
            }

            return std::make_tuple(503u, "text/plain", std::move(error_msg), nullptr);
        } catch (const std::exception& e)
        {
            // Refer above comment about not including exception messages in the response
            std::string error_msg = "Error occurred while pushing payload to queue";
            LOG(ERROR) << error_msg << ": " << e.what();
            return std::make_tuple(500u, "text/plain", error_msg, nullptr);
        }
    };

    request_handler_fn_t live_parser = [this](const tcp_endpoint_t& tcp_endpoint, const request_t& request) {
        if (!m_server->is_running())
        {
            std::string error_msg = "Source server is not running";
            return std::make_tuple(500u, "text/plain", error_msg, nullptr);
        }

        return std::make_tuple(200u, "text/plain", std::string(), nullptr);
    };

    request_handler_fn_t ready_parser = [this](const tcp_endpoint_t& tcp_endpoint, const request_t& request) {
        if (!m_server->is_running())
        {
            std::string error_msg = "Source server is not running";
            return std::make_tuple(500u, "text/plain", error_msg, nullptr);
        }

        if (m_queue_cnt < m_max_queue_size)
        {
            return std::make_tuple(200u, "text/plain", std::string(), nullptr);
        }

        std::string error_msg = "HTTP payload queue is full or unavailable to accept new values";
        return std::make_tuple(503u, "text/plain", std::move(error_msg), nullptr);
    };

    std::vector<HttpEndpoint> endpoints;
    endpoints.emplace_back(parser, std::move(endpoint), std::move(method));
    endpoints.emplace_back(live_parser, std::move(live_endpoint), std::move(live_method));
    endpoints.emplace_back(ready_parser, std::move(ready_endpoint), std::move(ready_method));

    m_server = std::make_unique<HttpServer>(
        std::move(endpoints), std::move(bind_address), port, num_server_threads, max_payload_size, request_timeout);
}

template <typename OutputT>
HttpServerSourceStage<OutputT>::subscriber_fn_t HttpServerSourceStage<OutputT>::build()
{
    return [this](rxcpp::subscriber<source_type_t> subscriber) -> void {
        try
        {
            m_server->start();
            this->source_generator(subscriber);
        } catch (const SourceStageStopAfter& e)
        {
            DLOG(INFO) << "Completed after emitting " << m_records_emitted << " records";
        } catch (const std::exception& e)
        {
            LOG(ERROR) << "Encountered error while listening for incoming HTTP requests: " << e.what() << std::endl;
            subscriber.on_error(std::make_exception_ptr(e));
            return;
        }
        subscriber.on_completed();
        this->close();
    };
}

template <typename OutputT>
void HttpServerSourceStage<OutputT>::source_generator(
    rxcpp::subscriber<HttpServerSourceStage::source_type_t> subscriber)
{
    // only check if the server is running when the queue is empty, allowing all queued messages to be processed prior
    // to shutting down
    bool server_running = true;
    bool queue_closed   = false;
    while (subscriber.is_subscribed() && server_running && !queue_closed)
    {
        table_t table_ptr{nullptr};
        auto queue_status = m_queue.try_pop(table_ptr);
        if (queue_status == boost::fibers::channel_op_status::success)
        {
            --m_queue_cnt;
            // NOLINTNEXTLINE(clang-diagnostic-unused-value)
            DCHECK_NOTNULL(table_ptr);
            try
            {
                auto message     = MessageMeta::create_from_cpp(std::move(table_ptr->first), 0);
                auto num_records = message->count();

                // When OutputT is MessageMeta, we just swap the pointers
                std::shared_ptr<OutputT> out_message{nullptr};
                make_output_message(message, m_task.get(), std::move(table_ptr->second), out_message);

                subscriber.on_next(std::move(out_message));
                m_records_emitted += num_records;
            } catch (const std::exception& e)
            {
                LOG(ERROR) << "Error occurred converting HTTP payload to Dataframe: " << e.what();
            }

            if (m_stop_after > 0 && m_records_emitted >= m_stop_after)
            {
                throw SourceStageStopAfter();
            }
        }
        else if (queue_status == boost::fibers::channel_op_status::empty)
        {
            // if the queue is empty, maybe it's because our server is not running
            server_running = m_server->is_running();

            if (server_running)
            {
                // Sleep when there are no messages
                boost::this_fiber::sleep_for(m_sleep_time);
            }
        }
        else if (queue_status == boost::fibers::channel_op_status::closed)
        {
            queue_closed = true;
        }
        else
        {
            std::string error_msg{"Unknown queue status: " + std::to_string(static_cast<int>(queue_status))};
            LOG(ERROR) << error_msg;
            throw std::runtime_error(error_msg);
        }
    }
}

template <typename OutputT>
void HttpServerSourceStage<OutputT>::close()
{
    if (m_server)
    {
        m_server->stop();  // this is a no-op if the server is not running
        m_server.reset();
    }
    m_queue.close();
}

/** @} */  // end of group
}  // namespace morpheus
