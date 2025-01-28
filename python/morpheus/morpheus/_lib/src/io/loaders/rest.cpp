/*
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

#include "morpheus/io/loaders/rest.hpp"

#include "morpheus/messages/control.hpp"
#include "morpheus/messages/meta.hpp"

#include <boost/asio.hpp>
#include <boost/asio/any_io_executor.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/core/buffers_to_string.hpp>
#include <boost/beast/core/error.hpp>
#include <boost/beast/core/flat_buffer.hpp>
#include <boost/beast/core/multi_buffer.hpp>
#include <boost/beast/core/rate_policy.hpp>
#include <boost/beast/core/tcp_stream.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/http/basic_dynamic_body.hpp>
#include <boost/beast/http/dynamic_body.hpp>
#include <boost/beast/http/error.hpp>
#include <boost/beast/http/field.hpp>
#include <boost/beast/http/fields.hpp>
#include <boost/beast/http/message.hpp>
#include <boost/beast/http/status.hpp>
#include <boost/beast/http/string_body.hpp>
#include <boost/beast/http/verb.hpp>
#include <boost/beast/version.hpp>
#include <boost/intrusive/detail/algo_type.hpp>
#include <boost/intrusive/link_mode.hpp>
#include <boost/system/detail/errc.hpp>
#include <boost/system/detail/error_code.hpp>
#include <glog/logging.h>
#include <nlohmann/json.hpp>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pymrc/utilities/object_cache.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace py = pybind11;

namespace beast = boost::beast;
namespace http  = beast::http;
namespace net   = boost::asio;

namespace {
void extract_query_fields(nlohmann::json& query,
                          std::string& method,
                          std::string& endpoint,
                          std::string& port,
                          std::string& http_version,
                          std::string& content_type,
                          std::string& body,
                          nlohmann::json& params,
                          nlohmann::json& x_headers)
{
    method = query.value("method", "GET");
    std::transform(method.begin(), method.end(), method.begin(), ::toupper);

    endpoint = query.value("endpoint", "");
    if (endpoint.empty())
    {
        throw std::runtime_error("'REST loader' receives query with empty endpoint");
    }

    port         = query.value("port", "80");
    http_version = query.value("http_version", "1.1");
    content_type = query.value("content_type", "");
    body         = query.value("body", "");
    params       = query.value("params", nlohmann::json());
    x_headers    = query.value("x-headers", nlohmann::json());
}

http::verb get_http_verb(const std::string& method)
{
    http::verb verb;
    if (method == "GET")
    {
        verb = http::verb::get;
    }
    else if (method == "POST")
    {
        verb = http::verb::post;
    }
    else
    {
        throw std::runtime_error("'REST Loader' receives method not supported: " + method);
    }
    return verb;
}

void construct_request(http::request<http::string_body>& request,
                       const std::string& method,
                       const std::string& host,
                       const std::string& target,
                       const std::string& query,
                       const std::string& http_version,
                       const std::string& content_type,
                       const std::string& body,
                       const nlohmann::json& x_headers)
{
    auto verb = get_http_verb(method);
    request.method(verb);
    request.target(target + query);

    // Supported HTTP versions are HTTP/1.1 and HTTP/1.0 :
    //   - Currently Beast does not support HTTP/2.0 (see
    //     https://www.boost.org/doc/libs/1_82_0/libs/beast/doc/html/beast/design_choices/faq.html)
    //   - The default value of http_version is 1.1 if not specified within REST loader's config, since Beast sets
    //     to 1.1 by default.
    //   - Beast provides http client examples compatible with HTTP/1.0 (see
    //     https://www.boost.org/doc/libs/1_82_0/libs/beast/example/http/client/sync/http_client_sync.cpp). REST loader
    //     implements the http client in the same way as Beast's example, providing support to HTTP/1.0 if explicitly
    //     specified in the config.
    if (http_version != "1.1" && http_version != "1.0")
    {
        throw std::runtime_error("'REST Loader' received http version not supported: " + http_version);
    }
    unsigned version = http_version == "1.1" ? 11 : 10;
    request.version(version);

    request.set(http::field::host, host);
    request.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
    if (verb == http::verb::post)
    {
        request.set(http::field::content_type, content_type);
        request.body() = body;
        request.prepare_payload();
    }
    for (auto& x_header : x_headers.items())
    {
        request.insert(x_header.key(), x_header.value().dump());
    }
}

void get_response_with_retry(http::request<http::string_body>& request,
                             const std::string& port,
                             http::response<http::dynamic_body>& response,
                             int max_retry,
                             int retry_interval_milliseconds)
{
    net::io_context ioc;
    net::ip::tcp::resolver resolver(ioc);
    beast::tcp_stream stream(ioc);
    auto const endpoint_results = resolver.resolve(std::string(request[http::field::host]), port);
    while (max_retry > 0)
    {
        stream.connect(endpoint_results);
        http::write(stream, request);
        beast::flat_buffer buffer;
        http::read(stream, buffer, response);
        beast::error_code ec;
        stream.socket().shutdown(net::ip::tcp::socket::shutdown_both, ec);
        if (ec && ec != beast::errc::not_connected)
        {
            throw beast::system_error{ec};
        }
        unsigned status = response.result_int();
        // Retry only if status is 503 Service Unavailable / 504 Gateway Timeout
        if (status != (unsigned)http::status::service_unavailable && status != (unsigned)http::status::gateway_timeout)
        {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(retry_interval_milliseconds));
        retry_interval_milliseconds *= 2;
        max_retry--;
    }
    if (max_retry == 0)
    {
        throw std::runtime_error("'REST loader' reaches max_retry count");
    }
}

py::dict convert_url_query_to_dict(const py::str& raw_query)
{
    // Convert query string (param1=true&param2=false) into Python dict ({"param1": "true", "param2": "false"}) for
    // encoding
    py::dict params_dict;
    if (py::len(raw_query) > 0)
    {
        py::list split_by_amp = raw_query.attr("split")("&");

        for (auto& query_kv : split_by_amp)
        {
            py::list split_by_eq = query_kv.attr("split")("=");
            if (py::len(split_by_eq) < 2)
            {
                throw std::runtime_error("'REST Loader' failed to parse URL query: " + raw_query.cast<std::string>());
            }
            params_dict[split_by_eq[0]] = split_by_eq[1];
        }
    }
    return params_dict;
}

void parse_endpoint(std::string& endpoint, std::string& host, std::string& target, std::string& query)
{
    py::gil_scoped_acquire gil;
    py::module_ urllib = py::module::import("urllib.parse");

    // Following the syntax specifications in RFC 1808, urlparse recognizes a netloc only if it is properly
    // introduced by ‘//’. Otherwise the input is presumed to be a relative URL and thus to start with a path component
    // ref: https://docs.python.org/3/library/urllib.parse.html
    // Workaround here is to prepend "http://" if endpoint does not already include one
    if (endpoint.find("//") == std::string::npos)
    {
        endpoint = "http://" + endpoint;
    }

    py::object result = urllib.attr("urlparse")(endpoint);

    // If not present in URL:
    //  - host: None
    //  - path/query: Empty Python String
    host = result.attr("hostname").is_none() ? "" : result.attr("hostname").cast<std::string>();

    // Encode URL target
    target = urllib.attr("quote")(result.attr("path")).cast<std::string>();
    target = target.empty() ? "/" : target;

    py::str raw_query = result.attr("query");
    py::dict query_pydict;
    query_pydict = convert_url_query_to_dict(raw_query);

    // Encode URL query
    query = py::len(query_pydict) == 0 ? "" : "?" + urllib.attr("urlencode")(query_pydict).cast<std::string>();
}

void parse_params(nlohmann::json& params, std::string& query)
{
    py::gil_scoped_acquire gil;
    py::module_ urllib = py::module::import("urllib.parse");

    py::dict query_pydict;
    for (auto& param : params.items())
    {
        query_pydict[py::str(param.key())] = py::str(param.value());
    }
    query = py::len(query_pydict) == 0 ? "" : "?" + urllib.attr("urlencode")(query_pydict).cast<std::string>();
}

void get_response_from_endpoint(http::response<http::dynamic_body>& response,
                                std::string& method,
                                std::string& host,
                                std::string& target,
                                std::string& query,
                                std::string& port,
                                std::string& http_version,
                                std::string& content_type,
                                std::string& body,
                                nlohmann::json& x_headers,
                                int max_retry,
                                int retry_interval_milliseconds)
{
    http::request<http::string_body> request;

    construct_request(request, method, host, target, query, http_version, content_type, body, x_headers);
    get_response_with_retry(request, port, response, max_retry, retry_interval_milliseconds);
}

void create_dataframe_from_response(py::object& dataframe,
                                    py::module_& mod_cudf,
                                    http::response<http::dynamic_body>& response,
                                    const std::string& strategy)
{
    std::string df_json_str = beast::buffers_to_string(response.body().data());

    // When calling cudf.read_json() with engine='cudf', it expects an array object as input.
    // The workaround here is to add square brackets if the original data is not represented as an array.
    // Submitted an issue to cudf: https://github.com/rapidsai/cudf/issues/13527.
    if (!df_json_str.empty() && !(df_json_str.front() == '[' && df_json_str.back() == ']'))
    {
        df_json_str = "[" + df_json_str + "]";
    }

    {
        py::gil_scoped_acquire gil;
        auto current_df = mod_cudf.attr("DataFrame")();
        current_df      = mod_cudf.attr("read_json")(py::str(df_json_str), py::str("cudf"));
        if (dataframe.is_none())
        {
            dataframe = current_df;
        }
        else if (strategy == "aggregate")
        {
            py::list args;
            args.attr("append")(dataframe);
            args.attr("append")(current_df);
            dataframe = mod_cudf.attr("concat")(args);
        }
    }  // release GIL
}

void create_dataframe_from_query(py::object& dataframe,
                                 py::module_& mod_cudf,
                                 nlohmann::json& query,
                                 int max_retry,
                                 int retry_interval_milliseconds,
                                 std::string& strategy)
{
    std::string method;
    std::string endpoint;
    std::string port;
    std::string http_version;
    std::string content_type;
    std::string body;
    nlohmann::json params;
    nlohmann::json x_headers;
    extract_query_fields(query, method, endpoint, port, http_version, content_type, body, params, x_headers);

    std::string host;
    std::string target;
    std::string queries;
    parse_endpoint(endpoint, host, target, queries);

    if (params.empty())
    {
        http::response<http::dynamic_body> response;
        get_response_from_endpoint(response,
                                   method,
                                   host,
                                   target,
                                   queries,
                                   port,
                                   http_version,
                                   content_type,
                                   body,
                                   x_headers,
                                   max_retry,
                                   retry_interval_milliseconds);
        create_dataframe_from_response(dataframe, mod_cudf, response, strategy);
    }
    else
    {
        // For each set of param, send a separate request
        for (auto& param : params)
        {
            parse_params(param, queries);
            http::response<http::dynamic_body> response;
            get_response_from_endpoint(response,
                                       method,
                                       host,
                                       target,
                                       queries,
                                       port,
                                       http_version,
                                       content_type,
                                       body,
                                       x_headers,
                                       max_retry,
                                       retry_interval_milliseconds);
            create_dataframe_from_response(dataframe, mod_cudf, response, strategy);
        }
    }
}
}  // namespace

namespace morpheus {
RESTDataLoader::RESTDataLoader(nlohmann::json config) : Loader(config) {}

std::shared_ptr<ControlMessage> RESTDataLoader::load(std::shared_ptr<ControlMessage> message, nlohmann::json task)
{
    VLOG(30) << "Called RESTDataLoader::load()";

    py::module_ mod_cudf;
    py::object dataframe;

    {
        py::gil_scoped_acquire gil;

        dataframe          = py::none();
        auto& cache_handle = mrc::pymrc::PythonObjectCache::get_handle();
        mod_cudf           = cache_handle.get_module("cudf");
    }  // release GIL

    try
    {
        auto conf = this->config();

        int max_retry = conf.value("max_retry", 3);
        if (max_retry < 0)
        {
            throw std::runtime_error("'REST Loader' receives invalid max_retry value: " + std::to_string(max_retry));
        }

        int retry_interval_milliseconds = conf.value("retry_interval_milliseconds", 1000);
        if (retry_interval_milliseconds < 0)
        {
            throw std::runtime_error("'REST Loader' receives invalid retry_interval_milliseconds value: " +
                                     std::to_string(retry_interval_milliseconds));
        }

        if (!task["queries"].is_array() or task.empty())
        {
            throw std::runtime_error("'REST Loader' control message specified no queries to load");
        }

        std::string strategy = task.value("strategy", "aggregate");
        if (strategy != "aggregate")
        {
            throw std::runtime_error("Only 'aggregate' strategy is currently supported");
        }

        auto queries = task["queries"];
        for (auto& query : queries)
        {
            create_dataframe_from_query(dataframe, mod_cudf, query, max_retry, retry_interval_milliseconds, strategy);
        }

        {
            py::gil_scoped_acquire gil;
            message->payload(MessageMeta::create_from_python(std::move(dataframe)));
        }  // release GIL
    } catch (...)
    {
        py::gil_scoped_acquire gil;
        auto _handle = dataframe.release();
        _handle.dec_ref();

        _handle = mod_cudf.release();
        _handle.dec_ref();

        throw;
    }

    return message;
}
}  // namespace morpheus
