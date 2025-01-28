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

#include "./include/py_llm_engine.hpp"
#include "./include/py_llm_node.hpp"
#include "./include/py_llm_node_base.hpp"
#include "./include/py_llm_task_handler.hpp"
#include "morpheus_llm/llm/input_map.hpp"
#include "morpheus_llm/llm/llm_context.hpp"
#include "morpheus_llm/llm/llm_engine.hpp"
#include "morpheus_llm/llm/llm_node.hpp"
#include "morpheus_llm/llm/llm_node_base.hpp"
#include "morpheus_llm/llm/llm_node_runner.hpp"
#include "morpheus_llm/llm/llm_task.hpp"
#include "morpheus_llm/llm/llm_task_handler.hpp"
#include "morpheus_llm/pybind11/input_map.hpp"  // IWYU pragma: keep
#include "py_llm_engine_stage.hpp"
#include "py_llm_lambda_node.hpp"

#include "morpheus/messages/control.hpp"  // IWYU pragma: keep
#include "morpheus/pybind11/json.hpp"     // IWYU pragma: keep
#include "morpheus/utilities/json_types.hpp"
#include "morpheus/version.hpp"

#include <mrc/segment/object.hpp>  // for Object, ObjectProperties
#include <mrc/utils/string_utils.hpp>
#include <nlohmann/detail/exceptions.hpp>  // for nlohmann::detail::out_of_range
#include <nlohmann/json.hpp>               // for basic_json
#include <pybind11/functional.h>           // IWYU pragma: keep
#include <pybind11/pybind11.h>  // for arg, init, class_, module_, str_attr_accessor, PYBIND11_MODULE, pybind11
#include <pybind11/stl.h>       // IWYU pragma: keep
#include <pymrc/coro.hpp>       // IWYU pragma: keep
#include <pymrc/utils.hpp>      // for pymrc::import

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
// IWYU pragma: no_include "morpheus_llm/llm/fwd.hpp"
// IWYU pragma: no_include <pybind11/attr.h>
// IWYU pragma: no_include <pybind11/pytypes.h>

namespace morpheus::llm {
namespace py = pybind11;

PYBIND11_MODULE(llm, _module)
{
    _module.doc() = R"pbdoc(
        -----------------------
        .. currentmodule:: morpheus_llm.llm
        .. autosummary::
           :toctree: _generate

        )pbdoc";

    // Import the mrc coro module
    mrc::pymrc::import(_module, "mrc.core.coro");

    // Import the messages module
    mrc::pymrc::import(_module, "morpheus._lib.messages");

    py::class_<InputMap>(_module, "InputMap")
        .def(py::init<>())
        .def(py::init<std::string, std::string>(), py::arg("external_name"), py::arg("internal_name"))
        .def_readwrite("external_name",
                       &InputMap::external_name,
                       "The name of node that will be mapped to this input. Use a leading '/' to indicate it is a "
                       "sibling node otherwise it will be treated as a parent node. Can also specify a specific node "
                       "output such as '/sibling_node/output1' to map the output 'output1' of 'sibling_node' to this "
                       "input. Can also use a wild card such as '/sibling_node/\\*' to match all internal node names")
        .def_readwrite(
            "internal_name",
            &InputMap::internal_name,
            "The internal node name that the external node maps to. Must match an input returned from "
            "`get_input_names()` of the desired node. Defaults to '-' which is a placeholder for the "
            "default input of the node. Use a wildcard '\\*' to match all inputs of the node (Must also use a "
            "wild card on the external mapping).");

    py::class_<LLMTask>(_module, "LLMTask")
        .def(py::init<>())
        .def(py::init([](std::string task_type, py::dict task_dict) {
                 return LLMTask(std::move(task_type), mrc::pymrc::cast_from_pyobject(task_dict));
             }),
             py::arg("task_type"),
             py::arg("task_dict"))
        .def_readonly("task_type", &LLMTask::task_type)
        .def(
            "__getitem__",
            [](const LLMTask& self, const std::string& key) {
                try
                {
                    return mrc::pymrc::cast_from_json(self.get(key));
                } catch (const std::out_of_range&)
                {
                    throw py::key_error("key '" + key + "' does not exist");
                }
            },
            py::arg("key"))
        .def(
            "__setitem__",
            [](LLMTask& self, const std::string& key, py::object value) {
                try
                {
                    // Convert to C++ nholman object
                    return self.set(key, mrc::pymrc::cast_from_pyobject(std::move(value)));
                } catch (const std::out_of_range&)
                {
                    throw py::key_error("key '" + key + "' does not exist");
                }
            },
            py::arg("key"),
            py::arg("value"))
        .def("__len__", &LLMTask::size)
        .def(
            "get",
            [](const LLMTask& self, const std::string& key) {
                try
                {
                    return mrc::pymrc::cast_from_json(self.get(key));
                } catch (const nlohmann::detail::out_of_range&)
                {
                    throw py::key_error("key '" + key + "' does not exist");
                }
            },
            py::arg("key"))
        .def(
            "get",
            [](const LLMTask& self, const std::string& key, py::object default_value) {
                try
                {
                    return mrc::pymrc::cast_from_json(self.get(key));
                } catch (const nlohmann::detail::out_of_range&)
                {
                    return default_value;
                }
            },
            py::arg("key"),
            py::arg("default_value"));
    // .def(
    //     "__iter__",
    //     [](const LLMTask& self) {
    //         return py::detail::make_iterator_impl<iterator_key_access<typename Iterator>,
    //                                               return_value_policy Policy,
    //                                               typename Iterator,
    //                                               typename Sentinel,
    //                                               typename ValueType>(self.task_dict.items().begin(),
    //                                                                   self.task_dict.items().end());
    //         self.task_dict.items().begin().begin() return py::make_key_iterator(map.begin(), map.end());
    //     },
    //     py::keep_alive<0, 1>())
    // .def(
    //     "items",
    //     [](const LLMTask& self) {
    //         return py::make_iterator(self.begin(), self.end());
    //     },
    //     py::keep_alive<0, 1>())
    // .def(
    //     "values",
    //     [](const LLMTask& map) {
    //         return py::make_value_iterator(self., self.end());
    //     },
    //     py::keep_alive<0, 1>());

    py::class_<LLMContext, std::shared_ptr<LLMContext>>(_module, "LLMContext")
        .def(py::init<>())
        .def(py::init<std::shared_ptr<LLMContext>, std::string, input_mappings_t>(),
             py::arg("prent"),
             py::arg("name"),
             py::arg("inputs"))
        .def(py::init<LLMTask, std::shared_ptr<ControlMessage>>(), py::arg("task"), py::arg("message"))
        .def_property_readonly("name", &LLMContext::name)
        .def_property_readonly("full_name", &LLMContext::full_name)
        .def_property_readonly("view_outputs", &LLMContext::view_outputs)
        .def_property_readonly("input_map", &LLMContext::input_map)
        .def_property_readonly("parent", &LLMContext::parent)
        .def("task", &LLMContext::task)
        .def("message", &LLMContext::message)
        .def("get_input", py::overload_cast<>(&LLMContext::get_input, py::const_))
        .def("get_input",
             py::overload_cast<const std::string&>(&LLMContext::get_input, py::const_),
             py::arg("node_name"))
        .def("get_inputs", &LLMContext::get_inputs)
        .def("set_output", py::overload_cast<utilities::json_t>(&LLMContext::set_output), py::arg("outputs"))
        .def("set_output",
             py::overload_cast<const std::string&, utilities::json_t>(&LLMContext::set_output),
             py::arg("output_name"),
             py::arg("output"))
        .def("push", &LLMContext::push, py::arg("name"), py::arg("inputs"));

    py::class_<LLMNodeBase, PyLLMNodeBase<>, std::shared_ptr<LLMNodeBase>>(_module, "LLMNodeBase")
        .def(py::init_alias<>())
        .def("get_input_names",
             &LLMNodeBase::get_input_names,
             R"pbdoc(
                Get the input names for the node.

                Returns
                -------
                list[str]
                    The input names for the node
             )pbdoc")
        .def("execute",
             &LLMNodeBase::execute,
             py::arg("context"),
             R"pbdoc(
                Execute the current node with the given `context` instance.

                All inputs for the given node should be fetched from the context, typically by calling either
                `context.get_inputs` to fetch all inputs as a `dict`, or `context.get_input` to fetch a specific input.

                Similarly the output of the node is written to the context using `context.set_output`.

                Parameters
                ----------
                context : `morpheus._lib.llm.LLMContext`
                    Context instance to use for the execution

            )pbdoc");

    py::class_<LLMNodeRunner, std::shared_ptr<LLMNodeRunner>>(_module, "LLMNodeRunner")
        .def_property_readonly("inputs", &LLMNodeRunner::inputs)
        .def_property_readonly("name", &LLMNodeRunner::name)
        .def_property_readonly("parent_input_names", &LLMNodeRunner::parent_input_names)
        .def_property_readonly("sibling_input_names", &LLMNodeRunner::sibling_input_names)
        .def("execute", &LLMNodeRunner::execute, py::arg("context"));

    py::class_<LLMNode, LLMNodeBase, PyLLMNode<>, std::shared_ptr<LLMNode>>(_module, "LLMNode")
        .def(py::init_alias<>())
        .def(
            "add_node",
            [](LLMNode& self, std::string name, py::object inputs, std::shared_ptr<LLMNodeBase> node, bool is_output) {
                user_input_mappings_t converted_inputs;

                if (inputs.is_none())
                {
                    // Populate the inputs from the node input_names
                    for (const auto& single_input : node->get_input_names())
                    {
                        converted_inputs.emplace_back(single_input);
                    }
                }
                else
                {
                    converted_inputs = inputs.cast<user_input_mappings_t>();
                }

                return self.add_node(std::move(name), std::move(converted_inputs), std::move(node), is_output);
            },
            py::arg("name"),
            py::kw_only(),
            py::arg("inputs") = py::none(),
            py::arg("node"),
            py::arg("is_output") = false,
            R"pbdoc(
                Add an LLMNode to the current node.

                Parameters
                ----------
                name : str
                    The name of the node to add

                inputs : list[tuple[str, str]], optional
                    List of input mappings to use for the node, in the form of `[(external_name, internal_name), ...]`
                    If unspecified the node's input_names will be used.

                node : LLMNodeBase
                    The node to add

                is_output : bool, optional
                    Indicates if the node is an output node, by default False

            )pbdoc");

    py::class_<LLMTaskHandler, PyLLMTaskHandler, std::shared_ptr<LLMTaskHandler>>(
        _module, "LLMTaskHandler", "Acts as a sink for an `LLMEngine`, emitting results as a `ControlMessage`")
        .def(py::init<>())
        .def("get_input_names",
             &LLMTaskHandler::get_input_names,
             R"pbdoc(
                Get the input names for the task handler.

                Returns
                -------
                list[str]
                    The input names for the task handler.

             )pbdoc")
        .def("try_handle",
             &LLMTaskHandler::try_handle,
             py::arg("context"),
             R"pbdoc(
                Convert the given `context` into a list of `ControlMessage` instances.

                Parameters
                ----------
                context : `morpheus._lib.llm.LLMContext`
                    Context instance to use for the execution

                Returns
                -------
                Task[Optional[list[ControlMessage]]]

              )pbdoc");

    py::class_<LLMEngine, LLMNode, PyLLMEngine, std::shared_ptr<LLMEngine>>(_module, "LLMEngine")
        .def(py::init_alias<>())
        .def("add_task_handler", &LLMEngine::add_task_handler, py::arg("inputs"), py::arg("handler"))
        .def("run", &LLMEngine::run, py::arg("message"));

    py::class_<PyLLMLambdaNode, LLMNodeBase, std::shared_ptr<PyLLMLambdaNode>>(_module, "LLMLambdaNode")
        .def(py::init<>([](py::function fn) {
                 return std::make_shared<PyLLMLambdaNode>(std::move(fn));
             }),
             py::arg("fn"))
        .def("get_input_names", &PyLLMLambdaNode::get_input_names)
        .def("execute", &PyLLMLambdaNode::execute, py::arg("context"));

    py::class_<mrc::segment::Object<PyLLMEngineStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<PyLLMEngineStage>>>(
        _module, "LLMEngineStage", py::multiple_inheritance())
        .def(py::init<>(&PyLLMEngineStage::init), py::arg("builder"), py::arg("name"), py::arg("engine"));

    _module.attr("__version__") =
        MRC_CONCAT_STR(morpheus_VERSION_MAJOR << "." << morpheus_VERSION_MINOR << "." << morpheus_VERSION_PATCH);
}
}  // namespace morpheus::llm
