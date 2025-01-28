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

#include "morpheus/messages/control.hpp"

#include "morpheus/messages/memory/tensor_memory.hpp"  // for TensorMemory, TensorMemoryInterfaceProxy
#include "morpheus/messages/meta.hpp"                  // for MessageMeta, MessageMetaInterfaceProxy
#include "morpheus/types.hpp"                          // for TensorIndex

#include <boost/algorithm/string.hpp>  // for to_lower_copy
#include <glog/logging.h>              // for COMPACT_GOOGLE_LOG_INFO, LogMessage, VLOG
#include <nlohmann/json.hpp>           // for basic_json, json_ref, iter_impl, operator<<
#include <pybind11/chrono.h>           // IWYU pragma: keep
#include <pybind11/pybind11.h>         // for cast, object::cast
#include <pybind11/pytypes.h>          // for object, none, dict, isinstance, list, str, value_error, generic_item
#include <pybind11/stl.h>              // IWYU pragma: keep
#include <pymrc/utils.hpp>             // for cast_from_pyobject

#include <optional>   // for optional, nullopt
#include <ostream>    // for basic_ostream, operator<<
#include <regex>      // for regex_search, regex
#include <stdexcept>  // for runtime_error
#include <utility>    // for pair
// IWYU pragma: no_include <boost/iterator/iterator_facade.hpp>

namespace py = pybind11;
using namespace py::literals;

namespace morpheus {

const std::string ControlMessage::s_config_schema = R"()";

std::map<std::string, ControlMessageType> ControlMessage::s_task_type_map{{"inference", ControlMessageType::INFERENCE},
                                                                          {"none", ControlMessageType::NONE},
                                                                          {"training", ControlMessageType::TRAINING}};

ControlMessage::ControlMessage() : m_config({{"metadata", morpheus::utilities::json_t::object()}}), m_tasks({}) {}

ControlMessage::ControlMessage(const morpheus::utilities::json_t& _config) :
  m_config({{"metadata", morpheus::utilities::json_t::object()}}),
  m_tasks({})
{
    config(_config);
}

ControlMessage::ControlMessage(const ControlMessage& other)
{
    m_cm_type      = other.m_cm_type;
    m_payload      = other.m_payload;
    m_tensors      = other.m_tensors;
    m_tensor_count = other.m_tensor_count;

    m_config = other.m_config;
    m_tasks  = other.m_tasks;

    m_timestamps = other.m_timestamps;
}

const morpheus::utilities::json_t& ControlMessage::config() const
{
    return m_config;
}

void ControlMessage::add_task(const std::string& task_type, const morpheus::utilities::json_t& task)
{
    VLOG(20) << "Adding task of type " << task_type << " to control message" << task.dump(4);
    auto _task_type = to_task_type(task_type, false);

    if (_task_type != ControlMessageType::NONE)
    {
        auto current_task_type = this->task_type();
        if (current_task_type == ControlMessageType::NONE)
        {
            this->task_type(_task_type);
        }
        else if (current_task_type != _task_type)
        {
            throw std::runtime_error("Cannot mix different types of tasks on the same control message");
        }
    }

    m_tasks[task_type].push_back(task);
}

bool ControlMessage::has_task(const std::string& task_type) const
{
    return m_tasks.contains(task_type) && m_tasks.at(task_type).size() > 0;
}

const morpheus::utilities::json_t& ControlMessage::get_tasks() const
{
    return m_tasks;
}

std::vector<std::string> ControlMessage::list_metadata() const
{
    std::vector<std::string> key_list{};

    for (auto it = m_config["metadata"].begin(); it != m_config["metadata"].end(); ++it)
    {
        key_list.push_back(it.key());
    }

    return key_list;
}

void ControlMessage::set_metadata(const std::string& key, const morpheus::utilities::json_t& value)
{
    if (m_config["metadata"].contains(key))
    {
        VLOG(20) << "Overwriting metadata key " << key << " with value " << value;
    }

    m_config["metadata"][key] = value;
}

bool ControlMessage::has_metadata(const std::string& key) const
{
    return m_config["metadata"].contains(key);
}

morpheus::utilities::json_t ControlMessage::get_metadata() const
{
    auto metadata = m_config["metadata"];

    return metadata;
}

morpheus::utilities::json_t ControlMessage::get_metadata(const std::string& key, bool fail_on_nonexist) const
{
    // Assuming m_metadata is a std::map<std::string, nlohmann::json> storing metadata
    auto metadata = m_config["metadata"];
    auto it       = metadata.find(key);
    if (it != metadata.end())
    {
        return metadata.at(key);
    }
    else if (fail_on_nonexist)
    {
        throw std::runtime_error("Metadata key does not exist: " + key);
    }
    return {};
}

morpheus::utilities::json_t ControlMessage::remove_task(const std::string& task_type)
{
    auto& task_set = m_tasks.at(task_type);
    auto iter_task = task_set.begin();

    if (iter_task != task_set.end())
    {
        auto task = *iter_task;
        task_set.erase(iter_task);

        return task;
    }

    throw std::runtime_error("No tasks of type " + task_type + " found");
}

void ControlMessage::set_timestamp(const std::string& key, time_point_t timestamp_ns)
{
    // Insert or update the timestamp in the map
    m_timestamps[key] = timestamp_ns;
}

const std::map<std::string, time_point_t>& ControlMessage::get_timestamps() const
{
    return m_timestamps;
}

std::map<std::string, time_point_t> ControlMessage::filter_timestamp(const std::string& regex_filter)
{
    std::map<std::string, time_point_t> matching_timestamps;
    std::regex filter(regex_filter);

    for (const auto& [key, timestamp] : m_timestamps)
    {
        // Check if the key matches the regex
        if (std::regex_search(key, filter))
        {
            matching_timestamps[key] = timestamp;
        }
    }

    return matching_timestamps;
}

std::optional<time_point_t> ControlMessage::get_timestamp(const std::string& key, bool fail_if_nonexist)
{
    auto it = m_timestamps.find(key);
    if (it != m_timestamps.end())
    {
        return it->second;  // Return the found timestamp
    }
    else if (fail_if_nonexist)
    {
        throw std::runtime_error("Timestamp for the specified key does not exist.");
    }
    return std::nullopt;
}

void ControlMessage::config(const morpheus::utilities::json_t& config)
{
    if (config.contains("type"))
    {
        this->task_type(to_task_type(config.at("type").get<std::string>(), true));
    }

    if (config.contains("tasks"))
    {
        auto& tasks = config["tasks"];
        for (const auto& task : tasks)
        {
            add_task(task.at("type"), task.at("properties"));
        }
    }

    if (config.contains("metadata"))
    {
        auto& metadata = config["metadata"];
        for (auto it = metadata.begin(); it != metadata.end(); ++it)
        {
            set_metadata(it.key(), it.value());
        }
    }
}

std::shared_ptr<MessageMeta> ControlMessage::payload()
{
    return m_payload;
}

void ControlMessage::payload(const std::shared_ptr<MessageMeta>& payload)
{
    m_payload = payload;
}

std::shared_ptr<TensorMemory> ControlMessage::tensors()
{
    return m_tensors;
}

void ControlMessage::tensors(const std::shared_ptr<TensorMemory>& tensors)
{
    m_tensors      = tensors;
    m_tensor_count = tensors ? tensors->count : 0;
}

TensorIndex ControlMessage::tensor_count()
{
    return m_tensor_count;
}

ControlMessageType ControlMessage::task_type()
{
    return m_cm_type;
}

void ControlMessage::task_type(ControlMessageType type)
{
    m_cm_type = type;
}

ControlMessageType ControlMessage::to_task_type(const std::string& task_type, bool throw_on_error) const
{
    auto lower_task_type = boost::to_lower_copy(task_type);
    if (ControlMessage::s_task_type_map.contains(lower_task_type))
    {
        return ControlMessage::s_task_type_map.at(lower_task_type);
    }

    if (throw_on_error)
    {
        throw std::runtime_error("Invalid task type: " + task_type);
    }

    return ControlMessageType::NONE;
}

/*** Proxy Implementations ***/
std::shared_ptr<ControlMessage> ControlMessageProxy::create(py::object& config_or_message)
{
    if (config_or_message.is_none())
    {
        return std::make_shared<ControlMessage>();
    }

    if (py::isinstance<py::dict>(config_or_message))
    {
        return std::make_shared<ControlMessage>(mrc::pymrc::cast_from_pyobject(config_or_message));
    }

    // Assume we received an instance of the Python impl of ControlMessage object, as a Python bound instance of the C++
    // impl of the ControlMessage class would have invoked the shared_ptr<ControlMessage> overload of the create method
    py::dict config = config_or_message.attr("_export_config")();
    auto cm         = std::make_shared<ControlMessage>(mrc::pymrc::cast_from_pyobject(config));

    auto py_meta = config_or_message.attr("payload")();
    if (!py_meta.is_none())
    {
        cm->payload(MessageMetaInterfaceProxy::init_python_meta(py_meta));
    }

    auto py_tensors = config_or_message.attr("tensors")();
    if (!py_tensors.is_none())
    {
        auto count          = py_tensors.attr("count").cast<TensorIndex>();
        auto py_tensors_map = py_tensors.attr("get_tensors")();
        cm->tensors(TensorMemoryInterfaceProxy::init(count, py_tensors_map));
    }

    auto py_timestamps = config_or_message.attr("_timestamps");
    if (!py_timestamps.is_none())
    {
        auto timestamps_map = py_timestamps.cast<std::map<std::string, time_point_t>>();
        for (const auto& t : timestamps_map)
        {
            cm->set_timestamp(t.first, t.second);
        }
    }

    return cm;
}

std::shared_ptr<ControlMessage> ControlMessageProxy::create(std::shared_ptr<ControlMessage> other)
{
    return std::make_shared<ControlMessage>(*other);
}

std::shared_ptr<ControlMessage> ControlMessageProxy::copy(ControlMessage& self)
{
    return std::make_shared<ControlMessage>(self);
}

py::object ControlMessageProxy::get_metadata(ControlMessage& self,
                                             const py::object& key,
                                             pybind11::object default_value)
{
    if (key.is_none())
    {
        auto metadata = self.get_metadata();
        return cast_from_json(metadata);
    }

    auto value = self.get_metadata(py::cast<std::string>(key), false);
    if (value.empty())
    {
        return default_value;
    }

    return cast_from_json(value);
}

py::list ControlMessageProxy::list_metadata(ControlMessage& self)
{
    auto keys = self.list_metadata();
    py::list py_keys;
    for (const auto& key : keys)
    {
        py_keys.append(py::str(key));
    }
    return py_keys;
}

py::dict ControlMessageProxy::get_timestamps(ControlMessage& self)
{
    return py::cast(self.get_timestamps());
}

py::dict ControlMessageProxy::filter_timestamp(ControlMessage& self, const std::string& regex_filter)
{
    auto cpp_map = self.filter_timestamp(regex_filter);
    py::dict py_dict;
    for (const auto& [key, timestamp] : cpp_map)
    {
        // Directly use the timestamp as datetime.datetime in Python
        py_dict[py::str(key)] = timestamp;
    }
    return py_dict;
}

// Get a specific timestamp and return it as datetime.datetime or None
py::object ControlMessageProxy::get_timestamp(ControlMessage& self, const std::string& key, bool fail_if_nonexist)
{
    try
    {
        auto timestamp_opt = self.get_timestamp(key, fail_if_nonexist);
        if (timestamp_opt)
        {
            // Directly return the timestamp as datetime.datetime in Python
            return py::cast(*timestamp_opt);
        }

        return py::none();
    } catch (const std::runtime_error& e)
    {
        if (fail_if_nonexist)
        {
            throw py::value_error(e.what());
        }
        return py::none();
    }
}

// Set a timestamp using a datetime.datetime object from Python
void ControlMessageProxy::set_timestamp(ControlMessage& self, const std::string& key, py::object timestamp_ns)
{
    if (!py::isinstance<py::none>(timestamp_ns))
    {
        // Convert Python datetime.datetime to std::chrono::system_clock::time_point before setting
        auto _timestamp_ns = timestamp_ns.cast<time_point_t>();
        self.set_timestamp(key, _timestamp_ns);
    }
    else
    {
        throw std::runtime_error("Timestamp cannot be None");
    }
}

void ControlMessageProxy::payload_from_python_meta(ControlMessage& self, const pybind11::object& meta)
{
    self.payload(MessageMetaInterfaceProxy::init_python_meta(meta));
}

}  // namespace morpheus
