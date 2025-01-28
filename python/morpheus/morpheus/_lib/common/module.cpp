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

#include "morpheus/io/data_loader_registry.hpp"
#include "morpheus/io/deserializers.hpp"  // for read_file_to_df
#include "morpheus/io/loaders/file.hpp"
#include "morpheus/io/loaders/grpc.hpp"
#include "morpheus/io/loaders/payload.hpp"
#include "morpheus/io/loaders/rest.hpp"
#include "morpheus/io/serializers.hpp"
#include "morpheus/objects/dtype.hpp"  // for TypeId
#include "morpheus/objects/fiber_queue.hpp"
#include "morpheus/objects/file_types.hpp"  // for FileTypes, determine_file_type
#include "morpheus/objects/filter_source.hpp"
#include "morpheus/objects/tensor_object.hpp"  // for TensorObject
#include "morpheus/objects/wrapped_tensor.hpp"
#include "morpheus/utilities/http_server.hpp"
#include "morpheus/version.hpp"

#include <mrc/utils/string_utils.hpp>
#include <nlohmann/json.hpp>
#include <pybind11/attr.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>  // for return_value_policy::reference
// for pathlib.Path -> std::filesystem::path conversions
#include <indicators/color.hpp>
#include <indicators/font_style.hpp>
#include <pybind11/stl.h>             // IWYU pragma: keep
#include <pybind11/stl/filesystem.h>  // IWYU pragma: keep

#include <filesystem>  // for std::filesystem::path
#include <memory>
#include <sstream>
#include <string>

namespace morpheus {
namespace py = pybind11;

PYBIND11_MODULE(common, _module)
{
    _module.doc() = R"pbdoc(
        -----------------------
        .. currentmodule:: morpheus.common
        .. autosummary::
           :toctree: _generate
        )pbdoc";

    LoaderRegistry::register_factory_fn(
        "file",
        [](nlohmann::json config) {
            return std::make_unique<FileDataLoader>(config);
        },
        false);
    LoaderRegistry::register_factory_fn(
        "grpc",
        [](nlohmann::json config) {
            return std::make_unique<GRPCDataLoader>(config);
        },
        false);
    LoaderRegistry::register_factory_fn(
        "payload",
        [](nlohmann::json config) {
            return std::make_unique<PayloadDataLoader>(config);
        },
        false);
    LoaderRegistry::register_factory_fn(
        "rest",
        [](nlohmann::json config) {
            return std::make_unique<RESTDataLoader>(config);
        },
        false);

    py::class_<TensorObject>(_module, "Tensor")
        .def_property_readonly("__cuda_array_interface__", &TensorObjectInterfaceProxy::cuda_array_interface)
        // No need to keep_alive here since cupy arrays have an owner object
        .def("to_cupy", &TensorObjectInterfaceProxy::to_cupy)
        // Need to set keep_alive here to keep the cupy array alive as long as the Tensor is
        .def_static("from_cupy", &TensorObjectInterfaceProxy::from_cupy, py::keep_alive<0, 1>());

    py::class_<FiberQueue, std::shared_ptr<FiberQueue>>(_module, "FiberQueue")
        .def(py::init<>(&FiberQueueInterfaceProxy::init), py::arg("max_size"))
        .def("get", &FiberQueueInterfaceProxy::get, py::arg("block") = true, py::arg("timeout") = 0.0)
        .def("put", &FiberQueueInterfaceProxy::put, py::arg("item"), py::arg("block") = true, py::arg("timeout") = 0.0)
        .def("close", &FiberQueueInterfaceProxy::close)
        .def("is_closed", &FiberQueueInterfaceProxy::is_closed)
        .def("__enter__", &FiberQueueInterfaceProxy::enter, py::return_value_policy::reference)
        .def("__exit__", &FiberQueueInterfaceProxy::exit);

    py::enum_<TypeId>(_module, "TypeId", "Supported Morpheus types")
        .value("EMPTY", TypeId::EMPTY)
        .value("INT8", TypeId::INT8)
        .value("INT16", TypeId::INT16)
        .value("INT32", TypeId::INT32)
        .value("INT64", TypeId::INT64)
        .value("UINT8", TypeId::UINT8)
        .value("UINT16", TypeId::UINT16)
        .value("UINT32", TypeId::UINT32)
        .value("UINT64", TypeId::UINT64)
        .value("FLOAT32", TypeId::FLOAT32)
        .value("FLOAT64", TypeId::FLOAT64)
        .value("BOOL8", TypeId::BOOL8)
        .value("STRING", TypeId::STRING);

    py::enum_<FileTypes>(_module,
                         "FileTypes",
                         "The type of files that the `FileSourceStage` can read and `WriteToFileStage` can write. Use "
                         "'auto' to determine from the file extension.")
        .value("Auto", FileTypes::Auto)
        .value("JSON", FileTypes::JSON)
        .value("CSV", FileTypes::CSV)
        .value("PARQUET", FileTypes::PARQUET);

    _module.def("typeid_to_numpy_str", [](TypeId tid) {
        return DType(tid).type_str();
    });

    _module.def("typeid_is_fully_supported", [](TypeId tid) {
        return DType(tid).is_fully_supported();
    });

    _module.def(
        "determine_file_type", py::overload_cast<const std::string&>(&determine_file_type), py::arg("filename"));
    _module.def("determine_file_type",
                py::overload_cast<const std::filesystem::path&>(&determine_file_type),
                py::arg("filename"));
    _module.def("read_file_to_df", &read_file_to_df, py::arg("filename"), py::arg("file_type") = FileTypes::Auto);
    _module.def("write_df_to_file",
                &SerializersProxy::write_df_to_file,
                py::arg("df"),
                py::arg("filename"),
                py::arg("file_type") = FileTypes::Auto);

    py::enum_<FilterSource>(
        _module, "FilterSource", "Enum to indicate which source the FilterDetectionsStage should operate on.")
        .value("Auto", FilterSource::Auto)
        .value("TENSOR", FilterSource::TENSOR)
        .value("DATAFRAME", FilterSource::DATAFRAME);

    py::class_<HttpEndpoint, std::shared_ptr<HttpEndpoint>>(_module, "HttpEndpoint")
        .def(py::init<>(&HttpEndpointInterfaceProxy::init),
             py::arg("py_parse_fn"),
             py::arg("url"),
             py::arg("method"),
             py::arg("include_headers") = false);

    py::class_<HttpServer, std::shared_ptr<HttpServer>>(_module, "HttpServer")
        .def(py::init<>(&HttpServerInterfaceProxy::init),
             py::arg("endpoints"),
             py::arg("bind_address")     = "127.0.0.1",
             py::arg("port")             = 8080,
             py::arg("num_threads")      = 1,
             py::arg("max_payload_size") = DefaultMaxPayloadSize,
             py::arg("request_timeout")  = 30)
        .def("start", &HttpServerInterfaceProxy::start)
        .def("stop", &HttpServerInterfaceProxy::stop)
        .def("is_running", &HttpServerInterfaceProxy::is_running)
        .def("__enter__", &HttpServerInterfaceProxy::enter, py::return_value_policy::reference)
        .def("__exit__", &HttpServerInterfaceProxy::exit);

    py::enum_<indicators::Color>(_module, "IndicatorsTextColor")
        .value("grey", indicators::Color::grey)
        .value("red", indicators::Color::red)
        .value("green", indicators::Color::green)
        .value("yellow", indicators::Color::yellow)
        .value("blue", indicators::Color::blue)
        .value("magenta", indicators::Color::magenta)
        .value("cyan", indicators::Color::cyan)
        .value("white", indicators::Color::white)
        .value("unspecified", indicators::Color::unspecified);

    py::enum_<indicators::FontStyle>(_module, "IndicatorsFontStyle")
        .value("bold", indicators::FontStyle::bold)
        .value("dark", indicators::FontStyle::dark)
        .value("italic", indicators::FontStyle::italic)
        .value("underline", indicators::FontStyle::underline)
        .value("blink", indicators::FontStyle::blink)
        .value("reverse", indicators::FontStyle::reverse)
        .value("concealed", indicators::FontStyle::concealed)
        .value("crossed", indicators::FontStyle::crossed);

    _module.attr("__version__") =
        MRC_CONCAT_STR(morpheus_VERSION_MAJOR << "." << morpheus_VERSION_MINOR << "." << morpheus_VERSION_PATCH);
}
}  // namespace morpheus
