/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "morpheus/messages/control.hpp"                 // for ControlMessage
#include "morpheus/messages/meta.hpp"                    // for MessageMeta
#include "morpheus/objects/file_types.hpp"               // for FileTypes
#include "morpheus/stages/add_classification.hpp"        // for AddClassificationsStage, AddClassificationStageInter...
#include "morpheus/stages/add_scores.hpp"                // for AddScoresStage, AddScoresStageInterfaceProxy
#include "morpheus/stages/deserialize.hpp"               // for DeserializeStage, DeserializeStageInterfaceProxy
#include "morpheus/stages/file_source.hpp"               // for FileSourceStage, FileSourceStageInterfaceProxy
#include "morpheus/stages/filter_detections.hpp"         // for FilterDetectionsStage, FilterDetectionStageInterface...
#include "morpheus/stages/http_server_source_stage.hpp"  // for HttpServerSourceStage, HttpServerSourceStageInterfac...
#include "morpheus/stages/inference_client_stage.hpp"    // for InferenceClientStage, InferenceClientStageInterfaceP...
#include "morpheus/stages/kafka_source.hpp"              // for KafkaSourceStage, KafkaSourceStageInterfaceProxy
#include "morpheus/stages/monitor.hpp"                   // for MonitorStage, MonitorStageInterfaceProxy
#include "morpheus/stages/preallocate.hpp"               // for PreallocateStage, PreallocateStageInterfaceProxy
#include "morpheus/stages/preprocess_fil.hpp"            // for PreprocessFILStage, PreprocessFILStageInterfaceProxy
#include "morpheus/stages/preprocess_nlp.hpp"            // for PreprocessNLPStage, PreprocessNLPStageInterfaceProxy
#include "morpheus/stages/serialize.hpp"                 // for SerializeStage, SerializeStageInterfaceProxy
#include "morpheus/stages/write_to_file.hpp"             // for WriteToFileStage, WriteToFileStageInterfaceProxy
#include "morpheus/utilities/http_server.hpp"            // for DefaultMaxPayloadSize
#include "morpheus/version.hpp"                          // for morpheus_VERSION_MAJOR, morpheus_VERSION_MINOR, morp...

#include <indicators/color.hpp>       // for Color
#include <indicators/font_style.hpp>  // for FontStyle
#include <mrc/segment/builder.hpp>     // for Builder
#include <mrc/segment/object.hpp>      // for Object, ObjectProperties
#include <mrc/utils/string_utils.hpp>  // for MRC_CONCAT_STR
#include <pybind11/attr.h>             // for multiple_inheritance
#include <pybind11/functional.h>       // IWYU pragma: keep
#include <pybind11/pybind11.h>         // for arg, init, class_, module_, overload_cast, overload_...
#include <pybind11/pytypes.h>          // for none, dict, str_attr
#include <pybind11/stl.h>              // IWYU pragma: keep
#include <pybind11/stl/filesystem.h>   // IWYU pragma: keep
#include <pymrc/utils.hpp>             // for import, from_import
#include <rxcpp/rx.hpp>                // for trace_activity, decay_t

#include <filesystem>  // for path
#include <memory>      // for shared_ptr, allocator
#include <sstream>     // for operator<<, basic_ostringstream
#include <string>      // for string
#include <vector>      // for vector
namespace morpheus {
namespace py = pybind11;

PYBIND11_MODULE(stages, _module)
{
    _module.doc() = R"pbdoc(
        -----------------------
        .. currentmodule:: morpheus._lib.stages
        .. autosummary::
           :toctree: _generate

        )pbdoc";

    // Make sure to load mrc.core.segment to get ObjectProperties
    mrc::pymrc::import(_module, "mrc.core.segment");

    // Import the mrc coro module
    mrc::pymrc::import(_module, "mrc.core.coro");

    // Import enums which are used as default argument values all of which require using py::arg_v
    // ref https://pybind11.readthedocs.io/en/latest/advanced/functions.html#default-arguments-revisited
    mrc::pymrc::from_import(_module, "morpheus._lib.common", "FileTypes");
    mrc::pymrc::from_import(_module, "morpheus._lib.common", "IndicatorsFontStyle");
    mrc::pymrc::from_import(_module, "morpheus._lib.common", "IndicatorsTextColor");

    py::class_<mrc::segment::Object<AddClassificationsStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<AddClassificationsStage>>>(
        _module, "AddClassificationsStage", py::multiple_inheritance())
        .def(py::init<>(&AddClassificationStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("idx2label"),
             py::arg("threshold"));

    py::class_<mrc::segment::Object<AddScoresStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<AddScoresStage>>>(
        _module, "AddScoresStage", py::multiple_inheritance())
        .def(
            py::init<>(&AddScoresStageInterfaceProxy::init), py::arg("builder"), py::arg("name"), py::arg("idx2label"));

    py::class_<mrc::segment::Object<DeserializeStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<DeserializeStage>>>(
        _module, "DeserializeStage", py::multiple_inheritance())
        .def(py::init<>(&DeserializeStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("batch_size"),
             py::arg("ensure_sliceable_index") = true,
             py::arg("task_type")              = py::none(),
             py::arg("task_payload")           = py::none());

    py::class_<mrc::segment::Object<FileSourceStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<FileSourceStage>>>(
        _module, "FileSourceStage", py::multiple_inheritance())
        .def(py::init(py::overload_cast<mrc::segment::Builder&,
                                        const std::string&,
                                        std::string,
                                        int,
                                        bool,
                                        std::vector<std::string>,
                                        py::dict>(&FileSourceStageInterfaceProxy::init)),
             py::arg("builder"),
             py::arg("name"),
             py::arg("filename"),
             py::arg("repeat"),
             py::arg("filter_null"),
             py::arg("filter_null_columns"),
             py::arg("parser_kwargs"))
        .def(py::init(py::overload_cast<mrc::segment::Builder&,
                                        const std::string&,
                                        std::filesystem::path,
                                        int,
                                        bool,
                                        std::vector<std::string>,
                                        py::dict>(&FileSourceStageInterfaceProxy::init)),
             py::arg("builder"),
             py::arg("name"),
             py::arg("filename"),
             py::arg("repeat"),
             py::arg("filter_null"),
             py::arg("filter_null_columns"),
             py::arg("parser_kwargs"));

    py::class_<mrc::segment::Object<FilterDetectionsStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<FilterDetectionsStage>>>(
        _module, "FilterDetectionsStage", py::multiple_inheritance())
        .def(py::init<>(&FilterDetectionStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("threshold"),
             py::arg("copy"),
             py::arg("filter_source"),
             py::arg("field_name") = "probs");

    py::class_<mrc::segment::Object<InferenceClientStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<InferenceClientStage>>>(
        _module, "InferenceClientStage", py::multiple_inheritance())
        .def(py::init<>(&InferenceClientStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("server_url"),
             py::arg("model_name"),
             py::arg("needs_logits"),
             py::arg("force_convert_inputs"),
             py::arg("input_mapping")  = py::dict(),
             py::arg("output_mapping") = py::dict());

    py::class_<mrc::segment::Object<KafkaSourceStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<KafkaSourceStage>>>(
        _module, "KafkaSourceStage", py::multiple_inheritance())
        .def(py::init<>(&KafkaSourceStageInterfaceProxy::init_with_single_topic),
             py::arg("builder"),
             py::arg("name"),
             py::arg("max_batch_size"),
             py::arg("topic"),
             py::arg("batch_timeout_ms"),
             py::arg("config"),
             py::arg("disable_commits")       = false,
             py::arg("disable_pre_filtering") = false,
             py::arg("stop_after")            = 0,
             py::arg("async_commits")         = true,
             py::arg("oauth_callback")        = py::none())
        .def(py::init<>(&KafkaSourceStageInterfaceProxy::init_with_multiple_topics),
             py::arg("builder"),
             py::arg("name"),
             py::arg("max_batch_size"),
             py::arg("topics"),
             py::arg("batch_timeout_ms"),
             py::arg("config"),
             py::arg("disable_commits")       = false,
             py::arg("disable_pre_filtering") = false,
             py::arg("stop_after")            = 0,
             py::arg("async_commits")         = true,
             py::arg("oauth_callback")        = py::none());

    py::class_<mrc::segment::Object<MonitorStage<MessageMeta>>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<MonitorStage<MessageMeta>>>>(
        _module, "MonitorMessageMetaStage", py::multiple_inheritance())
        .def(py::init<>(&MonitorStageInterfaceProxy<MessageMeta>::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("description"),
             py::arg("unit") = "messages",
             py::arg_v("text_color", indicators::Color::cyan, "IndicatorsTextColor.cyan"),
             py::arg_v("font_style", indicators::FontStyle::bold, "IndicatorsFontStyle.bold"),
             py::arg("determine_count_fn") = py::none());

    py::class_<mrc::segment::Object<MonitorStage<ControlMessage>>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<MonitorStage<ControlMessage>>>>(
        _module, "MonitorControlMessageStage", py::multiple_inheritance())
        .def(py::init<>(&MonitorStageInterfaceProxy<ControlMessage>::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("description"),
             py::arg("unit") = "messages",
             py::arg_v("text_color", indicators::Color::cyan, "IndicatorsTextColor.cyan"),
             py::arg_v("font_style", indicators::FontStyle::bold, "IndicatorsFontStyle.bold"),
             py::arg("determine_count_fn") = py::none());

    py::class_<mrc::segment::Object<PreallocateStage<ControlMessage>>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<PreallocateStage<ControlMessage>>>>(
        _module, "PreallocateControlMessageStage", py::multiple_inheritance())
        .def(py::init<>(&PreallocateStageInterfaceProxy<ControlMessage>::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("needed_columns"));

    py::class_<mrc::segment::Object<PreallocateStage<MessageMeta>>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<PreallocateStage<MessageMeta>>>>(
        _module, "PreallocateMessageMetaStage", py::multiple_inheritance())
        .def(py::init<>(&PreallocateStageInterfaceProxy<MessageMeta>::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("needed_columns"));

    py::class_<mrc::segment::Object<PreprocessFILStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<PreprocessFILStage>>>(
        _module, "PreprocessFILStage", py::multiple_inheritance())
        .def(py::init<>(&PreprocessFILStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("features"));

    py::class_<mrc::segment::Object<PreprocessNLPStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<PreprocessNLPStage>>>(
        _module, "PreprocessNLPStage", py::multiple_inheritance())
        .def(py::init<>(&PreprocessNLPStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("vocab_hash_file"),
             py::arg("sequence_length"),
             py::arg("truncation"),
             py::arg("do_lower_case"),
             py::arg("add_special_token"),
             py::arg("stride"),
             py::arg("column"));

    py::class_<mrc::segment::Object<HttpServerSourceStage<MessageMeta>>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<HttpServerSourceStage<MessageMeta>>>>(
        _module, "HttpServerMessageMetaSourceStage", py::multiple_inheritance())
        .def(py::init<>(&HttpServerSourceStageInterfaceProxy::init_meta),
             py::arg("builder"),
             py::arg("name"),
             py::arg("bind_address")       = "127.0.0.1",
             py::arg("port")               = 8080,
             py::arg("endpoint")           = "/message",
             py::arg("live_endpoint")      = "/live",
             py::arg("ready_endpoint")     = "/ready",
             py::arg("method")             = "POST",
             py::arg("live_method")        = "GET",
             py::arg("ready_method")       = "GET",
             py::arg("accept_status")      = 201u,
             py::arg("sleep_time")         = 0.1f,
             py::arg("queue_timeout")      = 5,
             py::arg("max_queue_size")     = 1024,
             py::arg("num_server_threads") = 1,
             py::arg("max_payload_size")   = DefaultMaxPayloadSize,
             py::arg("request_timeout")    = 30,
             py::arg("lines")              = false,
             py::arg("stop_after")         = 0);

    py::class_<mrc::segment::Object<HttpServerSourceStage<ControlMessage>>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<HttpServerSourceStage<ControlMessage>>>>(
        _module, "HttpServerControlMessageSourceStage", py::multiple_inheritance())
        .def(py::init<>(&HttpServerSourceStageInterfaceProxy::init_cm),
             py::arg("builder"),
             py::arg("name"),
             py::arg("bind_address")       = "127.0.0.1",
             py::arg("port")               = 8080,
             py::arg("endpoint")           = "/message",
             py::arg("live_endpoint")      = "/live",
             py::arg("ready_endpoint")     = "/ready",
             py::arg("method")             = "POST",
             py::arg("live_method")        = "GET",
             py::arg("ready_method")       = "GET",
             py::arg("accept_status")      = 201u,
             py::arg("sleep_time")         = 0.1f,
             py::arg("queue_timeout")      = 5,
             py::arg("max_queue_size")     = 1024,
             py::arg("num_server_threads") = 1,
             py::arg("max_payload_size")   = DefaultMaxPayloadSize,
             py::arg("request_timeout")    = 30,
             py::arg("lines")              = false,
             py::arg("stop_after")         = 0,
             py::arg("task_type")          = py::none(),
             py::arg("task_payload")       = py::none());

    py::class_<mrc::segment::Object<SerializeStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<SerializeStage>>>(
        _module, "SerializeStage", py::multiple_inheritance())
        .def(py::init<>(&SerializeStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("include"),
             py::arg("exclude"),
             py::arg("fixed_columns") = true);

    py::class_<mrc::segment::Object<WriteToFileStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<WriteToFileStage>>>(
        _module, "WriteToFileStage", py::multiple_inheritance())
        .def(py::init<>(&WriteToFileStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("filename"),
             py::arg("mode") = "w",
             py::arg_v("file_type", FileTypes::Auto, "FileTypes.Auto"),
             py::arg("include_index_col") = true,
             py::arg("flush")             = false);

    _module.attr("__version__") =
        MRC_CONCAT_STR(morpheus_VERSION_MAJOR << "." << morpheus_VERSION_MINOR << "." << morpheus_VERSION_PATCH);
}
}  // namespace morpheus
