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
#include "morpheus/messages/control.hpp"
#include "morpheus/messages/memory/inference_memory.hpp"
#include "morpheus/messages/memory/inference_memory_fil.hpp"
#include "morpheus/messages/memory/inference_memory_nlp.hpp"
#include "morpheus/messages/memory/response_memory.hpp"
#include "morpheus/messages/memory/response_memory_probs.hpp"
#include "morpheus/messages/memory/tensor_memory.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/raw_packet.hpp"
#include "morpheus/objects/data_table.hpp"
#include "morpheus/objects/mutable_table_ctx_mgr.hpp"
#include "morpheus/pybind11/json.hpp"         // IWYU pragma: keep
#include "morpheus/utilities/json_types.hpp"  // for json_t
#include "morpheus/utilities/string_util.hpp"
#include "morpheus/version.hpp"

#include <glog/logging.h>  // for COMPACT_GOOGLE_LOG_INFO, LogMessage, VLOG
#include <mrc/edge/edge_connector.hpp>
#include <nlohmann/json.hpp>      // for basic_json
#include <pybind11/functional.h>  // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep
#include <pymrc/node.hpp>  // IWYU pragma: keep
#include <pymrc/port_builders.hpp>
#include <pymrc/utilities/object_wrappers.hpp>
#include <pymrc/utils.hpp>  // for pymrc::import
#include <rxcpp/rx.hpp>

#include <cstddef>  // for size_t
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>     // IWYU pragma: keep
#include <typeinfo>  // for type_info
#include <utility>   // for index_sequence, make_index_sequence
#include <vector>
// For some reason IWYU thinks the variant header is needed for tuple, and that the array header is needed for
// tuple_element
// IWYU pragma: no_include <array>
// IWYU pragma: no_include <variant>

namespace morpheus {

namespace fs = std::filesystem;
namespace py = pybind11;

template <typename FirstT, typename SecondT>
void reg_converter()
{
    mrc::edge::EdgeConnector<std::shared_ptr<FirstT>, std::shared_ptr<SecondT>>::register_converter();
}

template <typename T>
void reg_py_type_helper()
{
    // Register the port util
    mrc::pymrc::PortBuilderUtil::register_port_util<std::shared_ptr<T>>();

    // Register conversion to and from python
    mrc::edge::EdgeConnector<std::shared_ptr<T>, mrc::pymrc::PyObjectHolder>::register_converter();
    mrc::edge::EdgeConnector<mrc::pymrc::PyObjectHolder, std::shared_ptr<T>>::register_converter();
}

template <typename TupleT, std::size_t I>
void do_register_tuple_index()
{
    static constexpr std::size_t LeftIndex  = I / std::tuple_size<TupleT>::value;
    static constexpr std::size_t RightIndex = I % std::tuple_size<TupleT>::value;

    using left_t  = typename std::tuple_element<LeftIndex, TupleT>::type;
    using right_t = typename std::tuple_element<RightIndex, TupleT>::type;

    // Only register if one of the types is a subclass of the other
    if constexpr (!std::is_same_v<left_t, right_t> && std::is_base_of_v<right_t, left_t>)
    {
        // Print the registration
        VLOG(20) << "[Type Registration]: Registering: " << typeid(left_t).name() << " -> " << typeid(right_t).name();
        reg_converter<left_t, right_t>();
    }
    else
    {
        VLOG(20) << "[Type Registration]: Skipping: " << typeid(left_t).name() << " -> " << typeid(right_t).name();
    }
};

template <typename TupleT, std::size_t... Is>
void register_tuple_index(std::index_sequence<Is...> /*unused*/)
{
    (do_register_tuple_index<TupleT, Is>(), ...);
}

template <typename... TypesT>
void register_permutations()
{
    register_tuple_index<std::tuple<TypesT...>>(std::make_index_sequence<(sizeof...(TypesT)) * (sizeof...(TypesT))>());
}

PYBIND11_MODULE(messages, _module)
{
    _module.doc() = R"pbdoc(
        -----------------------
        .. currentmodule:: morpheus.messages
        .. autosummary::
           :toctree: _generate

        )pbdoc";

    mrc::pymrc::import(_module, "cupy");  // It should be safe to import cupy in CPU only mode
    mrc::pymrc::import(_module, "morpheus._lib.common");

    // Required for SegmentObject
    mrc::pymrc::import(_module, "mrc.core.node");

    // Allows python objects to keep DataTable objects alive
    py::class_<IDataTable, std::shared_ptr<IDataTable>>(_module, "DataTable");

    // Add type registrations for all our common types
    reg_py_type_helper<ControlMessage>();
    reg_py_type_helper<MessageMeta>();

    // Tensor Memory classes
    py::class_<TensorMemory, std::shared_ptr<TensorMemory>>(_module, "TensorMemory")
        .def(py::init<>(&TensorMemoryInterfaceProxy::init),
             py::kw_only(),
             py::arg("count"),
             py::arg("tensors") = py::none())
        .def_readonly("count", &TensorMemory::count)
        .def_property_readonly("tensor_names", &TensorMemoryInterfaceProxy::tensor_names_getter)
        .def("has_tensor", &TensorMemoryInterfaceProxy::has_tensor)
        .def("get_tensors", &TensorMemoryInterfaceProxy::get_tensors, py::return_value_policy::move)
        .def("set_tensors", &TensorMemoryInterfaceProxy::set_tensors, py::arg("tensors"))
        .def("get_tensor", &TensorMemoryInterfaceProxy::get_tensor, py::arg("name"), py::return_value_policy::move)
        .def("set_tensor", &TensorMemoryInterfaceProxy::set_tensor, py::arg("name"), py::arg("tensor"));

    py::class_<InferenceMemory, TensorMemory, std::shared_ptr<InferenceMemory>>(_module, "InferenceMemory")
        .def(py::init<>(&InferenceMemoryInterfaceProxy::init),
             py::kw_only(),
             py::arg("count"),
             py::arg("tensors") = py::none())
        .def("get_input", &InferenceMemoryInterfaceProxy::get_tensor, py::arg("name"), py::return_value_policy::move)
        .def("set_input", &InferenceMemoryInterfaceProxy::set_tensor, py::arg("name"), py::arg("tensor"));

    py::class_<InferenceMemoryFIL, InferenceMemory, std::shared_ptr<InferenceMemoryFIL>>(_module, "InferenceMemoryFIL")
        .def(py::init<>(&InferenceMemoryFILInterfaceProxy::init),
             py::kw_only(),
             py::arg("count"),
             py::arg("input__0"),
             py::arg("seq_ids"))
        .def_property("input__0",
                      &InferenceMemoryFILInterfaceProxy::get_input__0,
                      &InferenceMemoryFILInterfaceProxy::set_input__0)
        .def_property(
            "seq_ids", &InferenceMemoryFILInterfaceProxy::get_seq_ids, &InferenceMemoryFILInterfaceProxy::set_seq_ids);

    py::class_<InferenceMemoryNLP, InferenceMemory, std::shared_ptr<InferenceMemoryNLP>>(_module, "InferenceMemoryNLP")
        .def(py::init<>(&InferenceMemoryNLPInterfaceProxy::init),
             py::kw_only(),
             py::arg("count"),
             py::arg("input_ids"),
             py::arg("input_mask"),
             py::arg("seq_ids"))
        .def_property("input_ids",
                      &InferenceMemoryNLPInterfaceProxy::get_input_ids,
                      &InferenceMemoryNLPInterfaceProxy::set_input_ids)
        .def_property("input_mask",
                      &InferenceMemoryNLPInterfaceProxy::get_input_mask,
                      &InferenceMemoryNLPInterfaceProxy::set_input_mask)
        .def_property(
            "seq_ids", &InferenceMemoryNLPInterfaceProxy::get_seq_ids, &InferenceMemoryNLPInterfaceProxy::set_seq_ids);

    py::class_<ResponseMemory, TensorMemory, std::shared_ptr<ResponseMemory>>(_module, "ResponseMemory")
        .def(py::init<>(&ResponseMemoryInterfaceProxy::init),
             py::kw_only(),
             py::arg("count"),
             py::arg("tensors") = py::none())
        .def("get_output", &ResponseMemoryInterfaceProxy::get_tensor, py::arg("name"), py::return_value_policy::move)
        .def("set_output", &ResponseMemoryInterfaceProxy::set_tensor, py::arg("name"), py::arg("tensor"));

    py::class_<ResponseMemoryProbs, ResponseMemory, std::shared_ptr<ResponseMemoryProbs>>(_module,
                                                                                          "ResponseMemoryProbs")
        .def(py::init<>(&ResponseMemoryProbsInterfaceProxy::init), py::kw_only(), py::arg("count"), py::arg("probs"))
        .def_property(
            "probs", &ResponseMemoryProbsInterfaceProxy::get_probs, &ResponseMemoryProbsInterfaceProxy::set_probs);

    // Context manager for Mutable Dataframes. Attempting to use it outside of a with block will raise an exception
    py::class_<MutableTableCtxMgr, std::shared_ptr<MutableTableCtxMgr>>(_module, "MutableTableCtxMgr")
        .def("__enter__", &MutableTableCtxMgr::enter, py::return_value_policy::reference)
        .def("__exit__", &MutableTableCtxMgr::exit)
        .def("__getattr__", &MutableTableCtxMgr::throw_usage_error)
        .def("__getitem__", &MutableTableCtxMgr::throw_usage_error)
        .def("__setattr__", &MutableTableCtxMgr::throw_usage_error)
        .def("__setitem__", &MutableTableCtxMgr::throw_usage_error);

    py::class_<MessageMeta, std::shared_ptr<MessageMeta>>(_module, "MessageMeta")
        .def(py::init<>(&MessageMetaInterfaceProxy::init_python), py::arg("df"))
        .def_property_readonly("count", &MessageMetaInterfaceProxy::count)
        .def_property_readonly("df", &MessageMetaInterfaceProxy::df_property, py::return_value_policy::move)
        .def("get_data",
             py::overload_cast<MessageMeta&>(&MessageMetaInterfaceProxy::get_data),
             py::return_value_policy::move)
        .def("get_data",
             py::overload_cast<MessageMeta&, std::string>(&MessageMetaInterfaceProxy::get_data),
             py::return_value_policy::move,
             py::arg("columns"))
        .def("get_data",
             py::overload_cast<MessageMeta&, std::vector<std::string>>(&MessageMetaInterfaceProxy::get_data),
             py::return_value_policy::move,
             py::arg("columns"))
        .def("get_data",
             py::overload_cast<MessageMeta&, pybind11::none>(&MessageMetaInterfaceProxy::get_data),
             py::return_value_policy::move,
             py::arg("columns"))
        .def("set_data", &MessageMetaInterfaceProxy::set_data, py::return_value_policy::move)
        .def("get_column_names", &MessageMetaInterfaceProxy::get_column_names)
        .def("copy_dataframe", &MessageMetaInterfaceProxy::get_data_frame, py::return_value_policy::move)
        .def("mutable_dataframe", &MessageMetaInterfaceProxy::mutable_dataframe, py::return_value_policy::move)
        .def("has_sliceable_index", &MessageMetaInterfaceProxy::has_sliceable_index)
        .def("ensure_sliceable_index", &MessageMetaInterfaceProxy::ensure_sliceable_index)
        .def("copy_ranges", &MessageMetaInterfaceProxy::copy_ranges, py::return_value_policy::move, py::arg("ranges"))
        .def("get_slice",
             &MessageMetaInterfaceProxy::get_slice,
             py::return_value_policy::move,
             py::arg("start"),
             py::arg("stop"))
        .def_static("make_from_file", &MessageMetaInterfaceProxy::init_cpp);

    py::enum_<ControlMessageType>(_module, "ControlMessageType")
        .value("INFERENCE", ControlMessageType::INFERENCE)
        .value("NONE", ControlMessageType::INFERENCE)
        .value("TRAINING", ControlMessageType::TRAINING);

    py::class_<ControlMessage, std::shared_ptr<ControlMessage>>(_module, "ControlMessage")
        .def(py::init<>())
        .def(py::init(py::overload_cast<std::shared_ptr<ControlMessage>>(&ControlMessageProxy::create)))
        .def(py::init(py::overload_cast<py::object&>(&ControlMessageProxy::create)))
        .def("add_task", &ControlMessage::add_task, py::arg("task_type"), py::arg("task"))
        .def(
            "config", py::overload_cast<const morpheus::utilities::json_t&>(&ControlMessage::config), py::arg("config"))
        .def("config", py::overload_cast<>(&ControlMessage::config, py::const_))
        .def("copy", &ControlMessageProxy::copy)
        .def("get_metadata",
             &ControlMessageProxy::get_metadata,
             py::arg("key")           = py::none(),
             py::arg("default_value") = py::none())
        .def("get_tasks", &ControlMessage::get_tasks)
        .def("filter_timestamp",
             py::overload_cast<ControlMessage&, const std::string&>(&ControlMessageProxy::filter_timestamp),
             "Retrieve timestamps matching a regex filter within a given group.",
             py::arg("regex_filter"))
        .def("get_timestamp",
             py::overload_cast<ControlMessage&, const std::string&, bool>(&ControlMessageProxy::get_timestamp),
             "Retrieve the timestamp for a given group and key. Returns None if the timestamp does not exist and "
             "fail_if_nonexist is False.",
             py::arg("key"),
             py::arg("fail_if_nonexist") = false)
        .def("get_timestamps", &ControlMessageProxy::get_timestamps)
        .def("set_timestamp",
             &ControlMessageProxy::set_timestamp,
             "Set a timestamp for a given key and group.",
             py::arg("key"),
             py::arg("timestamp"))
        .def("has_metadata", &ControlMessage::has_metadata, py::arg("key"))
        .def("has_task", &ControlMessage::has_task, py::arg("task_type"))
        .def("list_metadata", &ControlMessageProxy::list_metadata)
        .def("payload", pybind11::overload_cast<>(&ControlMessage::payload))
        .def("payload", pybind11::overload_cast<const std::shared_ptr<MessageMeta>&>(&ControlMessage::payload))
        .def(
            "payload",
            pybind11::overload_cast<ControlMessage&, const py::object&>(&ControlMessageProxy::payload_from_python_meta),
            py::arg("meta"))
        .def("tensors", pybind11::overload_cast<>(&ControlMessage::tensors))
        .def("tensors", pybind11::overload_cast<const std::shared_ptr<TensorMemory>&>(&ControlMessage::tensors))
        .def("tensor_count", &ControlMessage::tensor_count)
        .def("remove_task", &ControlMessage::remove_task, py::arg("task_type"))
        .def("set_metadata", &ControlMessage::set_metadata, py::arg("key"), py::arg("value"))
        .def("task_type", pybind11::overload_cast<>(&ControlMessage::task_type))
        .def(
            "task_type", pybind11::overload_cast<ControlMessageType>(&ControlMessage::task_type), py::arg("task_type"));

    py::class_<LoaderRegistry, std::shared_ptr<LoaderRegistry>>(_module, "DataLoaderRegistry")
        .def_static("contains", &LoaderRegistry::contains, py::arg("name"))
        .def_static("list", &LoaderRegistry::list)
        .def_static("register_loader",
                    &LoaderRegistryProxy::register_proxy_factory_fn,
                    py::arg("name"),
                    py::arg("loader"),
                    py::arg("throw_if_exists") = true)
        .def_static("unregister_loader",
                    &LoaderRegistry::unregister_factory_fn,
                    py::arg("name"),
                    py::arg("throw_if_not_exists") = true);

    py::class_<RawPacketMessage, std::shared_ptr<RawPacketMessage>>(_module, "RawPacketMessage")
        .def_property_readonly("num", &RawPacketMessage::count)
        .def_property_readonly("max_size", &RawPacketMessage::get_max_size)
        .def_property_readonly("gpu_mem", &RawPacketMessage::is_gpu_mem);

    _module.attr("__version__") =
        MORPHEUS_CONCAT_STR(morpheus_VERSION_MAJOR << "." << morpheus_VERSION_MINOR << "." << morpheus_VERSION_PATCH);
}
}  // namespace morpheus
