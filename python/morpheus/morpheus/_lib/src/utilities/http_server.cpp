/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.
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

#include "morpheus/utilities/http_server.hpp"

#include <boost/asio.hpp>  // for dispatch, make_address
#include <boost/asio/any_io_executor.hpp>
#include <boost/asio/basic_socket_acceptor.hpp>  // for basic_socket_acceptor<>::executor_type
#include <boost/asio/basic_stream_socket.hpp>    // for basic_stream_socket
#include <boost/asio/dispatch.hpp>
#include <boost/asio/ip/address.hpp>  // for address
#include <boost/asio/ip/tcp.hpp>      // for acceptor, endpoint, socket,
#include <boost/asio/socket_base.hpp>  // for socket_base::reuse_address, socket_base, socket_base::max_listen_connections
#include <boost/asio/strand.hpp>       // for strand, make_strand, operator==
#include <boost/beast/core.hpp>        // for bind_front_handler, error_code, flat_buffer, tcp_stream
#include <boost/beast/core/bind_handler.hpp>  // for bind_front_handler
#include <boost/beast/core/error.hpp>         // for error_code
#include <boost/beast/core/flat_buffer.hpp>   // for flat_buffer
#include <boost/beast/core/rate_policy.hpp>
#include <boost/beast/core/tcp_stream.hpp>  // for tcp_stream
#include <boost/beast/http.hpp>             // for read_async, response, write_async
#include <boost/beast/http/error.hpp>       // for error, error::end_of_stream
#include <boost/beast/http/field.hpp>       // for field, field::content_type
#include <boost/beast/http/fields.hpp>
#include <boost/beast/http/message.hpp>  // for message, response, request
#include <boost/beast/http/parser.hpp>   // for request_parser, parser
#include <boost/beast/http/status.hpp>   // for status, status::not_found
#include <boost/beast/http/verb.hpp>     // for verb, operator<<, verb::unknown
#include <boost/core/detail/string_view.hpp>
#include <glog/logging.h>     // for CHECK and LOG
#include <nlohmann/json.hpp>  // for basic_json, json_ref
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>  // IWYU pragma: keep
#include <pybind11/pytypes.h>
#include <pymrc/utilities/function_wrappers.hpp>  // for PyFuncWrapper
#include <pymrc/utils.hpp>                        // for cast_from_json

#include <algorithm>  // for max
#include <exception>  // for exception
#include <memory>
#include <ostream>      // needed for glog
#include <stdexcept>    // for runtime_error, length_error
#include <type_traits>  // indirectly used by pybind11 casting
#include <utility>      // for move
// IWYU pragma: no_include <boost/intrusive/detail/list_iterator.hpp>

// loosely based on the following examples:
// https://www.boost.org/doc/libs/1_74_0/libs/beast/example/http/server/async/http_server_async.cpp
// https://www.boost.org/doc/libs/1_74_0/libs/beast/example/advanced/server/advanced_server.cpp

namespace {
namespace beast = boost::beast;  // from <boost/beast.hpp>
namespace http  = beast::http;   // from <boost/beast/http.hpp>
namespace net   = boost::asio;   // from <boost/asio.hpp>

// from <boost/asio/ip/tcp.hpp>
using tcp = boost::asio::ip::tcp;  // NOLINT(readability-identifier-naming)
using namespace std::literals::chrono_literals;

class Session : public std::enable_shared_from_this<Session>
{
  public:
    Session(tcp::socket&& socket,
            std::vector<morpheus::HttpEndpoint>& endpoints,
            std::size_t max_payload_size,
            std::chrono::seconds timeout) :
      m_stream{std::move(socket)},
      m_endpoints{endpoints},
      m_max_payload_size{max_payload_size},
      m_timeout{timeout},
      m_on_complete_cb{nullptr}
    {}

    ~Session() = default;

    void run()
    {
        net::dispatch(m_stream.get_executor(), beast::bind_front_handler(&Session::do_read, shared_from_this()));
    }

  private:
    void do_read()
    {
        m_parser = std::make_unique<http::request_parser<http::string_body>>();
        m_parser->body_limit(m_max_payload_size);
        m_stream.expires_after(m_timeout);

        http::async_read(
            m_stream, m_buffer, *m_parser, beast::bind_front_handler(&Session::on_read, shared_from_this()));
    }

    void on_read(beast::error_code ec, std::size_t bytes_transferred)
    {
        if (ec == http::error::end_of_stream)
        {
            return do_close();
        }

        if (ec)
        {
            LOG(ERROR) << "Error reading request: " << ec.message();
            return;
        }

        // Release ownership of the parsed message and move it into the
        // handle_request method
        handle_request(m_parser->release());
    }

    void handle_request(http::request<http::string_body>&& request)
    {
        DLOG(INFO) << "Received request: " << request.method() << " : " << request.target();
        m_response         = std::make_unique<http::response<http::string_body>>();
        bool valid_request = false;

        for (const auto& endpoint : m_endpoints)
        {
            if (request.target() == endpoint.m_url && request.method() == endpoint.m_method)
            {
                valid_request = true;
                std::tuple<unsigned, std::string, std::string, morpheus::on_complete_cb_fn_t> parse_status;
                if (endpoint.m_request_handler != nullptr)
                {
                    parse_status = (*endpoint.m_request_handler)(m_stream.socket().remote_endpoint(), request);
                }
                else
                {
                    std::string body{request.body()};
                    parse_status = (*endpoint.m_parser)(body);
                }

                m_response->result(std::get<0>(parse_status));
                m_response->set(http::field::content_type, std::get<1>(parse_status));
                m_response->body() = std::get<2>(parse_status);
                m_on_complete_cb   = std::get<3>(parse_status);
                break;
            }
        }

        if (!valid_request)
        {
            m_response->result(http::status::not_found);
            m_response->set(http::field::content_type, "text/plain");
            m_response->body() = "not found";
        }

        try
        {
            DLOG(INFO) << "Response: " << m_response->result_int();
            m_response->keep_alive(request.keep_alive());
            m_response->prepare_payload();

            http::async_write(
                m_stream,
                *m_response,
                beast::bind_front_handler(&Session::on_write, shared_from_this(), m_response->need_eof()));
        } catch (const std::exception& e)
        {
            LOG(ERROR) << "Caught exception while writing response: " << e.what();
        }
    }

    void on_write(bool close, beast::error_code ec, std::size_t bytes_transferred)
    {
        if (ec)
        {
            LOG(ERROR) << "Error writing response: " << ec.message();
        }

        if (m_on_complete_cb)
        {
            try
            {
                m_on_complete_cb(ec);
            } catch (const std::exception& e)
            {
                LOG(ERROR) << "Caught exception while calling on_complete callback: " << e.what();
            } catch (...)
            {
                LOG(ERROR) << "Caught unknown exception while calling on_complete callback";
            }

            m_on_complete_cb = nullptr;
        }

        m_parser.reset(nullptr);
        m_response.reset(nullptr);

        if (close)
        {
            return do_close();
        }

        do_read();
    }

    void do_close()
    {
        beast::error_code ec;
        m_stream.socket().shutdown(tcp::socket::shutdown_send, ec);
    }

    beast::tcp_stream m_stream;
    beast::flat_buffer m_buffer;
    std::vector<morpheus::HttpEndpoint> m_endpoints;
    std::size_t m_max_payload_size;
    std::chrono::seconds m_timeout;

    // The response, and parser are all reset for each incoming request
    std::unique_ptr<http::request_parser<http::string_body>> m_parser;
    std::unique_ptr<http::response<http::string_body>> m_response;
    morpheus::on_complete_cb_fn_t m_on_complete_cb;
};

}  // namespace

namespace morpheus {

HttpServer::HttpServer(std::vector<HttpEndpoint> endpoints,
                       std::string bind_address,
                       unsigned short port,
                       unsigned short num_threads,
                       std::size_t max_payload_size,
                       std::chrono::seconds request_timeout) :
  m_endpoints(std::move(endpoints)),
  m_bind_address(std::move(bind_address)),
  m_port(port),
  m_num_threads(num_threads),
  m_request_timeout(request_timeout),
  m_max_payload_size(max_payload_size),
  m_io_context{m_num_threads},
  m_is_running{false}
{
    if (m_num_threads == 0)
    {
        throw std::runtime_error("num_threads must be greater than 0");
    }
}

void HttpServer::start_listener(std::binary_semaphore& listener_semaphore, std::binary_semaphore& started_semaphore)
{
    listener_semaphore.acquire();

    DCHECK(m_listener_threads.size() == 1 && m_listener_threads[0].get_id() == std::this_thread::get_id())
        << "start_listener must be called from the first thread in "
           "m_listener_threads";

    m_listener = std::make_shared<Listener>(
        m_io_context, m_bind_address, m_port, m_endpoints, m_max_payload_size, m_request_timeout);
    m_listener->run();

    for (auto i = 1; i < m_num_threads; ++i)
    {
        net::io_context& ioc = m_io_context;
        m_listener_threads.emplace_back([&ioc]() {
            ioc.run();
        });
    }

    m_is_running = true;
    started_semaphore.release();
    m_io_context.run();
}

void HttpServer::start()
{
    CHECK(!is_running()) << "HttpServer is already running";

    try
    {
        DLOG(INFO) << "Starting HttpServer on " << m_bind_address << ":" << m_port << " with " << m_num_threads
                   << " threads";
        m_listener_threads.reserve(m_num_threads);

        std::binary_semaphore listener_semaphore{0};
        std::binary_semaphore started_semaphore{0};
        m_listener_threads.emplace_back(
            &HttpServer::start_listener, this, std::ref(listener_semaphore), std::ref(started_semaphore));
        listener_semaphore.release();
        started_semaphore.acquire();
    } catch (const std::exception& e)
    {
        LOG(ERROR) << "Caught exception while starting HTTP server: " << e.what();
        stop();
    }
}

void HttpServer::stop()
{
    m_io_context.stop();
    while (!m_io_context.stopped())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    for (auto& t : m_listener_threads)
    {
        t.join();
    }

    m_listener_threads.clear();

    if (m_listener)
    {
        // io context stopped, so we can stop the listener
        m_listener->stop();
        m_listener.reset();
    }

    m_is_running = false;
}

bool HttpServer::is_running() const
{
    return m_is_running;
}

HttpServer::~HttpServer()
{
    try
    {
        if (is_running())
        {
            stop();
        }
    } catch (const std::exception& e)
    {
        LOG(ERROR) << "Caught exception while stopping HTTP server: " << e.what();
    }
}

utilities::json_t request_headers_to_json(const tcp_endpoint_t& tcp_endpoint, const request_t& request)
{
    morpheus::utilities::json_t headers{{"method", request.method_string()},
                                        {"endpoint", request.target()},
                                        {"remote_address", tcp_endpoint.address().to_string()},
                                        {"remote_port", tcp_endpoint.port()}};

    for (const auto& field : request)
    {
        headers[field.name_string()] = field.value();
    }

    return headers;
}

/****** HttpEndpointInterfaceProxy *************************/
using mrc::pymrc::PyFuncWrapper;
namespace py = pybind11;

std::shared_ptr<HttpEndpoint> HttpEndpointInterfaceProxy::init(pybind11::function py_parse_fn,
                                                               std::string url,
                                                               std::string method,
                                                               bool include_headers)
{
    auto wrapped_parse_fn                   = PyFuncWrapper(std::move(py_parse_fn));
    request_handler_fn_t request_handler_fn = [include_headers, wrapped_parse_fn = std::move(wrapped_parse_fn)](
                                                  const tcp_endpoint_t& tcp_endpoint, const request_t& request) {
        std::string body{request.body()};
        std::unique_ptr<utilities::json_t> headers{nullptr};
        if (include_headers)
        {
            headers = std::make_unique<utilities::json_t>(std::move(request_headers_to_json(tcp_endpoint, request)));
        }

        py::gil_scoped_acquire gil;
        auto py_payload = py::str(body);
        pybind11::tuple py_result;
        if (include_headers)
        {
            py::dict py_headers = mrc::pymrc::cast_from_json(*headers);
            py_result           = wrapped_parse_fn.operator()<py::tuple, py::str, py::dict>(py_payload, py_headers);
        }
        else
        {
            py_result = wrapped_parse_fn.operator()<py::tuple, py::str>(py_payload);
        }

        on_complete_cb_fn_t cb_fn{nullptr};
        if (!py_result[3].is_none())
        {
            auto py_cb_fn      = py_result[3].cast<py::function>();
            auto wrapped_cb_fn = PyFuncWrapper(std::move(py_cb_fn));

            cb_fn = [wrapped_cb_fn = std::move(wrapped_cb_fn)](const beast::error_code& ec) {
                py::gil_scoped_acquire gil;
                py::bool_ has_error = false;
                py::str error_msg;
                if (ec)
                {
                    has_error = true;
                    error_msg = ec.message();
                }

                wrapped_cb_fn.operator()<void, py::bool_, py::str>(has_error, error_msg);
            };
        }

        return std::make_tuple(py::cast<unsigned>(py_result[0]),
                               py::cast<std::string>(py_result[1]),
                               py::cast<std::string>(py_result[2]),
                               std::move(cb_fn));
    };

    return std::make_shared<HttpEndpoint>(std::move(request_handler_fn), std::move(url), method);
}

/****** HttpServerInterfaceProxy *************************/

std::shared_ptr<HttpServer> HttpServerInterfaceProxy::init(std::vector<HttpEndpoint> endpoints,
                                                           std::string bind_address,
                                                           unsigned short port,
                                                           unsigned short num_threads,
                                                           std::size_t max_payload_size,
                                                           int64_t request_timeout)
{
    return std::make_shared<HttpServer>(std::move(endpoints),
                                        std::move(bind_address),
                                        port,
                                        num_threads,
                                        max_payload_size,
                                        std::chrono::seconds(request_timeout));
}

void HttpServerInterfaceProxy::start(HttpServer& self)
{
    pybind11::gil_scoped_release release;
    self.start();
}

void HttpServerInterfaceProxy::stop(HttpServer& self)
{
    pybind11::gil_scoped_release release;
    self.stop();
}

bool HttpServerInterfaceProxy::is_running(const HttpServer& self)
{
    pybind11::gil_scoped_release release;
    return self.is_running();
}

HttpServer& HttpServerInterfaceProxy::enter(HttpServer& self)
{
    self.start();
    return self;
}

void HttpServerInterfaceProxy::exit(HttpServer& self,
                                    const pybind11::object& type,
                                    const pybind11::object& value,
                                    const pybind11::object& traceback)
{
    pybind11::gil_scoped_release release;
    self.stop();
}

HttpEndpoint::HttpEndpoint(std::shared_ptr<request_handler_fn_t>&& request_handler_fn,
                           std::shared_ptr<payload_parse_fn_t>&& payload_parse_fn,
                           std::string&& url,
                           const std::string& method) :
  m_request_handler{std::move(request_handler_fn)},
  m_parser{std::move(payload_parse_fn)},
  m_url{std::move(url)},
  m_method{http::string_to_verb(method)}
{
    DCHECK(m_request_handler != nullptr || m_parser != nullptr)
        << "Either request_handler_fn or payload_parse_fn must be provided";

    DCHECK(m_request_handler == nullptr || m_parser == nullptr)
        << "Only one of request_handler_fn or payload_parse_fn can be provided";

    if (m_method == http::verb::unknown)
    {
        throw std::runtime_error("Invalid method: " + method);
    }

    if (m_url.front() != '/')
    {
        m_url.insert(m_url.begin(), '/');
    }
}

HttpEndpoint::HttpEndpoint(request_handler_fn_t request_handler_fn, std::string&& url, const std::string& method) :
  HttpEndpoint{
      std::move(std::make_shared<request_handler_fn_t>(std::move(request_handler_fn))), nullptr, std::move(url), method}
{}

HttpEndpoint::HttpEndpoint(payload_parse_fn_t payload_parse_fn, std::string&& url, const std::string& method) :
  HttpEndpoint{
      nullptr, std::move(std::make_shared<payload_parse_fn_t>(std::move(payload_parse_fn))), std::move(url), method}
{}

Listener::Listener(net::io_context& io_context,
                   const std::string& bind_address,
                   unsigned short port,
                   std::vector<HttpEndpoint> endpoints,
                   std::size_t max_payload_size,
                   std::chrono::seconds request_timeout) :
  m_io_context{io_context},
  m_tcp_endpoint{net::ip::make_address(bind_address), port},
  m_acceptor{std::make_unique<tcp::acceptor>(net::make_strand(m_io_context))},
  m_endpoints{std::move(endpoints)},
  m_max_payload_size{max_payload_size},
  m_request_timeout{request_timeout},
  m_is_running{false}
{
    m_acceptor->open(m_tcp_endpoint.protocol());
    m_acceptor->set_option(net::socket_base::reuse_address(true));
    m_acceptor->bind(m_tcp_endpoint);
    m_acceptor->listen(net::socket_base::max_listen_connections);
}

void Listener::stop()
{
    m_acceptor->close();
    m_is_running = false;
    m_acceptor.reset();
    for (auto& endpoint : m_endpoints)
    {
        endpoint.m_parser.reset();
    }
}

void Listener::run()
{
    net::dispatch(m_acceptor->get_executor(),
                  beast::bind_front_handler(&Listener::do_accept, this->shared_from_this()));
    m_is_running = true;
}

bool Listener::is_running() const
{
    return m_is_running;
}

void Listener::do_accept()
{
    m_acceptor->async_accept(net::make_strand(m_io_context),
                             beast::bind_front_handler(&Listener::on_accept, this->shared_from_this()));
}

void Listener::on_accept(beast::error_code ec, tcp::socket socket)
{
    if (ec)
    {
        LOG(ERROR) << "Error accepting connection: " << ec.message();
    }
    else
    {
        std::make_shared<Session>(std::move(socket), m_endpoints, m_max_payload_size, m_request_timeout)->run();
    }

    do_accept();
}

}  // namespace morpheus
