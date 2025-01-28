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

#include "morpheus/export.h"                  // for exporting symbols
#include "morpheus/utilities/json_types.hpp"  // for json_t

#include <boost/asio/io_context.hpp>         // for io_context
#include <boost/asio/ip/tcp.hpp>             // for tcp, tcp::acceptor, tcp::endpoint, tcp::socket
#include <boost/beast/core/error.hpp>        // for error_code
#include <boost/beast/http/message.hpp>      // for request
#include <boost/beast/http/string_body.hpp>  // for string_body
#include <boost/beast/http/verb.hpp>         // for verb
#include <boost/system/detail/error_code.hpp>
#include <pybind11/pytypes.h>  // for pybind11::function

#include <atomic>      // for atomic
#include <chrono>      // for seconds
#include <cstddef>     // for size_t
#include <cstdint>     // for int64_t
#include <functional>  // for function
#include <memory>      // for shared_ptr & unique_ptr
#include <semaphore>   // for semaphore
#include <string>      // for string
#include <thread>      // for thread
#include <tuple>       // for make_tuple, tuple
#include <vector>      // for vector

namespace morpheus {
/**
 * @addtogroup objects
 * @{
 * @file
 */

using on_complete_cb_fn_t = std::function<void(const boost::system::error_code& /* error message */)>;

/**
 * @brief A tuple consisting of the HTTP status code, mime type to be used for the Content-Type header, and the body of
 * the response and optionally a callback function.
 */
using parse_status_t = std::tuple<unsigned /*http status code*/,
                                  std::string /* Content-Type of response */,
                                  std::string /* response body */,
                                  on_complete_cb_fn_t /* optional callback function, ignored if null */>;

// Note this is different than the http endpoint this represents the TCP connection
using tcp_endpoint_t = boost::asio::ip::tcp::endpoint;
using request_t      = boost::beast::http::request<boost::beast::http::string_body>;

/**
 * @brief A function that receives the TCP endpoint, and the request object. Returning an instance of `parse_status_t`.
 *
 * @details The function is expected to return a tuple conforming to `parse_status_t` consisting of the HTTP status
 * code, mime type value for the Content-Type header, body of the response and optionally a callback function.
 * If specified, the callback function which will be called once the response has been sent or failed to send, as
 * indicated by a `boost::system::error_code` reference passed to the function.
 *
 * Refer to https://www.boost.org/doc/libs/1_74_0/libs/system/doc/html/system.html#ref_class_error_code for more
 * information regarding `boost::system::error_code`.
 *
 * Note: This method is preferred over the payload_parse_fn_t as it provides access to the request headers.
 */
using request_handler_fn_t =
    std::function<parse_status_t(const tcp_endpoint_t& tcp_endpoint, const request_t& request)>;

/**
 * @brief A function that receives the post body and returns an HTTP status code, Content-Type string and body.
 *
 * @details The function is expected to return a tuple conforming to `parse_status_t` consisting of the HTTP status
 * code, mime type value for the Content-Type header, body of the response and optionally a callback function.
 * If specified, the callback function which will be called once the response has been sent or failed to send, as
 * indicated by a `boost::system::error_code` reference passed to the function.
 *
 * Refer to https://www.boost.org/doc/libs/1_74_0/libs/system/doc/html/system.html#ref_class_error_code for more
 * information regarding `boost::system::error_code`.
 *
 * Note: Ussage of this function is discouraged in favor of the request_handler_fn_t as it provides access to the
 * request headers, but remains available for compatibility. This method may be deprecated in the future.
 *
 */
using payload_parse_fn_t = std::function<parse_status_t(const std::string& body)>;

constexpr std::size_t DefaultMaxPayloadSize{1024 * 1024 * 10};  // 10MB

/**
 * @brief Convert the request headers to a JSON object.
 *
 * @param tcp_endpoint The TCP endpoint of the request.
 * @param request The request object.
 * @return The JSON object representing the request headers.
 */
utilities::json_t request_headers_to_json(const tcp_endpoint_t& tcp_endpoint, const request_t& request);

/**
 * @brief A struct that encapsulates the http endpoint attributes
 *
 * @details Constructed to be used in the HttpServer class as http endpoint configurations
 */
struct MORPHEUS_EXPORT HttpEndpoint
{
    HttpEndpoint(request_handler_fn_t request_handler_fn, std::string&& url, const std::string& method);
    HttpEndpoint(payload_parse_fn_t payload_parse_fn, std::string&& url, const std::string& method);

    std::shared_ptr<request_handler_fn_t> m_request_handler;
    std::shared_ptr<payload_parse_fn_t> m_parser;
    std::string m_url;
    boost::beast::http::verb m_method;

  private:
    HttpEndpoint(std::shared_ptr<request_handler_fn_t>&& request_handler_fn,
                 std::shared_ptr<payload_parse_fn_t>&& payload_parse_fn,
                 std::string&& url,
                 const std::string& method);
};

/**
 * @brief A class that listens for incoming HTTP requests.
 *
 * @details Constructed by the HttpServer class and should not be used directly.
 */
class MORPHEUS_EXPORT Listener : public std::enable_shared_from_this<Listener>
{
  public:
    Listener(boost::asio::io_context& io_context,
             const std::string& bind_address,
             unsigned short port,
             std::vector<HttpEndpoint> endpoints,
             std::size_t max_payload_size,
             std::chrono::seconds request_timeout);

    ~Listener() = default;

    void run();
    void stop();
    bool is_running() const;

  private:
    void do_accept();
    void on_accept(boost::beast::error_code ec, boost::asio::ip::tcp::socket socket);

    boost::asio::io_context& m_io_context;
    boost::asio::ip::tcp::endpoint m_tcp_endpoint;
    std::unique_ptr<boost::asio::ip::tcp::acceptor> m_acceptor;

    std::vector<HttpEndpoint> m_endpoints;
    std::size_t m_max_payload_size;
    std::chrono::seconds m_request_timeout;
    std::atomic<bool> m_is_running;
};

/**
 * @brief A simple HTTP server that listens for POST or PUT requests on a given endpoint.
 *
 * @details The server is started on a separate thread(s) and will call the provided payload_parse_fn_t
 *          function when an incoming request is received. The payload_parse_fn_t function is expected to
 *          return a tuple conforming to `parse_status_t` (ex: `std::make_tuple(200, "text/plain"s, "OK"s, nullptr)`).
 *
 * @param payload_parse_fn The function that will be called when a POST request is received.
 * @param bind_address The address to bind the server to.
 * @param port The port to bind the server to.
 * @param endpoint The endpoint to listen for POST requests on.
 * @param method The HTTP method to listen for.
 * @param num_threads The number of threads to use for the server.
 * @param max_payload_size The maximum size in bytes of the payload that the server will accept in a single request.
 * @param request_timeout The timeout for a request.
 */
class MORPHEUS_EXPORT HttpServer
{
  public:
    HttpServer(std::vector<HttpEndpoint> endpoints,
               std::string bind_address             = "127.0.0.1",
               unsigned short port                  = 8080,
               unsigned short num_threads           = 1,
               std::size_t max_payload_size         = DefaultMaxPayloadSize,
               std::chrono::seconds request_timeout = std::chrono::seconds(30));
    ~HttpServer();
    void start();
    void stop();
    bool is_running() const;

  private:
    void start_listener(std::binary_semaphore& listener_semaphore, std::binary_semaphore& started_semaphore);

    std::string m_bind_address;
    unsigned short m_port;
    std::vector<HttpEndpoint> m_endpoints;
    unsigned short m_num_threads;
    std::chrono::seconds m_request_timeout;
    std::size_t m_max_payload_size;
    std::vector<std::thread> m_listener_threads;
    boost::asio::io_context m_io_context;
    std::shared_ptr<Listener> m_listener;
    std::atomic<bool> m_is_running;
};

/****** HttpEndpointInterfaceProxy ************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MORPHEUS_EXPORT HttpEndpointInterfaceProxy
{
    static std::shared_ptr<HttpEndpoint> init(pybind11::function py_parse_fn,
                                              std::string m_url,
                                              std::string m_method,
                                              bool include_headers = false);
};

/****** HttpServerInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MORPHEUS_EXPORT HttpServerInterfaceProxy
{
    static std::shared_ptr<HttpServer> init(std::vector<HttpEndpoint> endpoints,
                                            std::string bind_address,
                                            unsigned short port,
                                            unsigned short num_threads,
                                            std::size_t max_payload_size,
                                            int64_t request_timeout);
    static void start(HttpServer& self);
    static void stop(HttpServer& self);
    static bool is_running(const HttpServer& self);

    // Context manager methods
    static HttpServer& enter(HttpServer& self);
    static void exit(HttpServer& self,
                     const pybind11::object& type,
                     const pybind11::object& value,
                     const pybind11::object& traceback);
};
}  // namespace morpheus
