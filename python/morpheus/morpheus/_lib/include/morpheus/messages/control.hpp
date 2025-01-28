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

#pragma once

#include "morpheus/export.h"                  // for MORPHEUS_EXPORT
#include "morpheus/messages/meta.hpp"         // for MessageMeta
#include "morpheus/types.hpp"
#include "morpheus/utilities/json_types.hpp"  // for json_t

#include <pybind11/pytypes.h>  // for object, dict, list
#include <pybind11/stl.h>      // IWYU pragma: keep

// for system_clock, time_point
#include <chrono>    // IWYU pragma: keep
#include <map>       // for map
#include <memory>    // for shared_ptr
#include <optional>  // for optional
#include <string>    // for string
#include <vector>    // for vector

// IWYU pragma: no_include <bits/chrono.h>

namespace morpheus {

enum class MORPHEUS_EXPORT ControlMessageType
{
    NONE,
    INFERENCE,
    TRAINING
};

class MORPHEUS_EXPORT TensorMemory;

// System-clock for better compatibility with pybind11/chrono
using time_point_t = std::chrono::time_point<std::chrono::system_clock>;

/**
 * @brief Class representing a control message for coordinating data processing tasks.
 *
 * This class contains configuration information, task definitions, and metadata, as well as a
 * pointer to an associated message payload. It provides methods for accessing and modifying these
 * elements of the control message.
 */
class MORPHEUS_EXPORT ControlMessage
{
  public:
    ControlMessage();
    explicit ControlMessage(const morpheus::utilities::json_t& config);

    ControlMessage(const ControlMessage& other);  // Copies config and metadata, but not payload

    /**
     * @brief Set the configuration object for the control message.
     * @param config A morpheus::utilities::json_t object containing configuration information.
     */
    void config(const morpheus::utilities::json_t& config);

    /**
     * @brief Get the configuration object for the control message.
     * @return A const reference to the morpheus::utilities::json_t object containing configuration information.
     */
    [[nodiscard]] const morpheus::utilities::json_t& config() const;

    /**
     * @brief Add a task of the given type to the control message.
     * @param task_type A string indicating the type of the task.
     * @param task A morpheus::utilities::json_t object describing the task.
     */
    void add_task(const std::string& task_type, const morpheus::utilities::json_t& task);

    /**
     * @brief Check if a task of the given type exists in the control message.
     * @param task_type A string indicating the type of the task.
     * @return True if a task of the given type exists, false otherwise.
     */
    [[nodiscard]] bool has_task(const std::string& task_type) const;

    /**
     * @brief Remove and return a task of the given type from the control message.
     * @param task_type A string indicating the type of the task.
     * @return A morpheus::utilities::json_t object describing the task.
     */
    morpheus::utilities::json_t remove_task(const std::string& task_type);

    /**
     * @brief Get the tasks for the control message.
     */
    [[nodiscard]] const morpheus::utilities::json_t& get_tasks() const;

    /**
     * @brief Add a key-value pair to the metadata for the control message.
     * @param key A string key for the metadata value.
     * @param value A morpheus::utilities::json_t object describing the metadata value.
     */
    void set_metadata(const std::string& key, const morpheus::utilities::json_t& value);

    /**
     * @brief Check if a metadata key exists in the control message.
     * @param key A string indicating the metadata key.
     * @return True if the metadata key exists, false otherwise.
     */
    [[nodiscard]] bool has_metadata(const std::string& key) const;

    /**
     * @brief Get the metadata for the control message.
     */
    [[nodiscard]] morpheus::utilities::json_t get_metadata() const;

    /**
     * @brief Get the metadata value for the given key from the control message.
     * If the key does not exist, the behavior depends on the fail_on_nonexist parameter.
     *
     * @param key A string indicating the metadata key.
     * @param fail_on_nonexist If true, throws an exception when the key does not exist.
     *                         If false, returns std::nullopt for non-existing keys.
     * @return An optional morpheus::utilities::json_t object describing the metadata value if it exists.
     */
    [[nodiscard]] morpheus::utilities::json_t get_metadata(const std::string& key, bool fail_on_nonexist = false) const;

    /**
     * @brief Lists all metadata keys currently stored in the control message.
     *
     * This method retrieves a list of all metadata keys present in the control message.
     * Metadata within a control message typically includes supplementary information
     * such as configuration settings, operational parameters, or annotations that
     * are not directly part of the message payload but are crucial for processing
     * or understanding the message.
     *
     * @return A std::vector<std::string> containing the keys of all metadata entries
     *         in the control message. If no metadata has been set, the returned vector
     *         will be empty.
     */
    [[nodiscard]] std::vector<std::string> list_metadata() const;

    /**
     * @brief Retrieves the current payload object of the control message.
     *
     * This method returns a shared pointer to the current payload object associated
     * with this control message. The payload object encapsulates metadata or data
     * specific to this message instance.
     *
     * @return A shared pointer to the MessageMeta instance representing the message payload.
     * @brief Get the payload object for the control message.
     * @param payload
     * A shared pointer to the message payload.
     */
    std::shared_ptr<MessageMeta> payload();

    /**
     * @brief Assigns a new payload object to the control message.
     *
     * Sets the payload of the control message to the provided MessageMeta instance.
     * The payload contains data or metadata pertinent to the message. Using a shared
     * pointer ensures that the payload is managed efficiently with automatic reference
     * counting.
     *
     * @param payload A shared pointer to the MessageMeta instance to be set as the new payload.
     */
    void payload(const std::shared_ptr<MessageMeta>& payload);

    /**
     * @brief Retrieves the tensor memory associated with the control message.
     *
     * This method returns a shared pointer to the TensorMemory object linked with
     * the control message, if any. TensorMemory typically contains or references
     * tensors or other large data blobs relevant to the message's purpose.
     *
     * @return A shared pointer to the TensorMemory instance associated with the message,
     *         or nullptr if no tensor memory is set.
     */
    std::shared_ptr<TensorMemory> tensors();

    /**
     * @brief Associates tensor memory with the control message.
     *
     * Sets the tensor memory for the control message to the provided TensorMemory instance.
     * This tensor memory can contain tensors or large data blobs pertinent to the message.
     * Utilizing a shared pointer facilitates efficient memory management through automatic
     * reference counting.
     *
     * @param tensor_memory A shared pointer to the TensorMemory instance to be associated
     *                      with the control message.
     */
    void tensors(const std::shared_ptr<TensorMemory>& tensor_memory);

    /**
     * @brief Get the length of tensors in the tensor memory.
     *
     * @return The length of tensors in the tensor memory.
     */
    TensorIndex tensor_count();

    /**
     * @brief Get the type of task associated with the control message.
     * @return An enum value indicating the task type.
     */
    ControlMessageType task_type();

    /**
     * @brief Set the task type for the control message
     * @param task_type
     * @return
     */
    void task_type(ControlMessageType task_type);

    /**
     * @brief Sets a timestamp for a specific key.
     *
     * This method stores a timestamp associated with a unique identifier,
     * If the key already exists, its timestamp will be updated to the new value.
     *
     * @param key The specific key for which the timestamp is to be set.
     * @param timestamp The timestamp to be associated with the key.
     */
    void set_timestamp(const std::string& key, time_point_t timestamp_ns);

    /**
     * @brief Retrieves the timestamp for a specific key.
     *
     * Attempts to find and return the timestamp associated with the specified key.
     * If the key does not exist, the method's behavior is determined by the fail_if_nonexist flag.
     *
     * @param key The specific key for which the timestamp is requested.
     * @param fail_if_nonexist If true, the method throws an exception if the timestamp doesn't exist.
     *                         If false, returns std::nullopt for non-existing timestamps.
     * @return An optional containing the timestamp if found, or std::nullopt
     * otherwise.
     */
    std::optional<time_point_t> get_timestamp(const std::string& key, bool fail_if_nonexist = false);

    /**
     * @brief Return a reference to the timestamps map
     *
     * @return A const map reference containing timestamps
     */
    const std::map<std::string, time_point_t>& get_timestamps() const;

    /**
     * @brief Retrieves timestamps for all keys that match a regex pattern.
     *
     * Searches for the specified for keys that match the provided regex filter and returns
     * a map of these keys and their associated timestamps.
     *
     * @param regex_filter A regular expression pattern that keys must match to be included in the result.
     * @return A map containing the matching key and their timestamps. The map will be empty if no matches are found.
     */
    std::map<std::string, time_point_t> filter_timestamp(const std::string& regex_filter);

  private:
    static const std::string s_config_schema;                          // NOLINT
    static std::map<std::string, ControlMessageType> s_task_type_map;  // NOLINT

    ControlMessageType to_task_type(const std::string& task_type, bool throw_on_error) const;

    ControlMessageType m_cm_type{ControlMessageType::NONE};
    std::shared_ptr<MessageMeta> m_payload{nullptr};
    std::shared_ptr<TensorMemory> m_tensors{nullptr};
    TensorIndex m_tensor_count{0};

    morpheus::utilities::json_t m_tasks{};
    morpheus::utilities::json_t m_config{};

    std::map<std::string, time_point_t> m_timestamps{};
};

struct MORPHEUS_EXPORT ControlMessageProxy
{
    /**
     * @brief Creates a new ControlMessage instance from either a Python instance of a ControlMessage or a configuration
     * dictionary.
     * @param config_or_message Either a Python instance of a ControlMessage or a dict representing the configuration
     * for the ControlMessage.
     * @return A shared_ptr to a newly created ControlMessage instance.
     */
    static std::shared_ptr<ControlMessage> create(pybind11::object& config_or_message);

    /**
     * @brief Creates a new ControlMessage instance as a copy of an existing one.
     * @param other A shared_ptr to another ControlMessage instance to copy.
     * @return A shared_ptr to the newly copied ControlMessage instance.
     */
    static std::shared_ptr<ControlMessage> create(std::shared_ptr<ControlMessage> other);

    /**
     * @brief Creates a deep copy of the ControlMessage instance.
     * @param self Reference to the underlying ControlMessage object.
     * @return A shared_ptr to the copied ControlMessage instance.
     */
    static std::shared_ptr<ControlMessage> copy(ControlMessage& self);

    /**
     * @brief Retrieves a metadata value by key, with an optional default value.
     *
     * @param self Reference to the underlying ControlMessage object.
     * @param key The key for the metadata entry. If not provided, retrieves all metadata.
     * @param default_value An optional default value to return if the key does not exist.
     * @return The value associated with the key, the default value if the key is not found, or all metadata if the key
     * is not provided.
     */
    static pybind11::object get_metadata(ControlMessage& self,
                                         const pybind11::object& key,
                                         pybind11::object default_value);

    /**
     * @brief Lists all metadata keys of the ControlMessage.
     * @param self Reference to the underlying ControlMessage object.
     * @return A pybind11::list containing all metadata keys.
     */
    static pybind11::list list_metadata(ControlMessage& self);

    /**
     * @brief Set the payload object given a Python instance of MessageMeta
     * @param meta
     */
    static void payload_from_python_meta(ControlMessage& self, const pybind11::object& meta);

    /**
     * @brief Sets a timestamp for a given key.
     * @param self Reference to the underlying ControlMessage object.
     * @param key The key associated with the timestamp.
     * @param timestamp A datetime.datetime object representing the timestamp.
     *
     * This method directly takes a datetime.datetime object from Python and sets the corresponding
     * std::chrono::system_clock::time_point for the specified key in the ControlMessage object.
     */
    static void set_timestamp(ControlMessage& self, const std::string& key, pybind11::object timestamp);

    /**
     * @brief Retrieves the timestamp for a specific key from the ControlMessage object.
     *
     * @param self Reference to the underlying ControlMessage object.
     * @param key The specific key for which the timestamp is requested.
     * @param fail_if_nonexist Determines the behavior when the requested timestamp does not exist.
     *                         If true, an exception is thrown. If false, py::none is returned.
     * @return A datetime.datetime object representing the timestamp if found, or py::none if not found
     *         and fail_if_nonexist is false.
     *
     * This method fetches the timestamp associated with the specified key and returns it as a
     * datetime.datetime object in Python. If the timestamp does not exist and fail_if_nonexist is true,
     * an exception is raised.
     */
    static pybind11::object get_timestamp(ControlMessage& self, const std::string& key, bool fail_if_nonexist = false);

    /**
     * @brief Return all timestamps
     *
     * @return A Python dictionary of timestamps
     */
    static pybind11::dict get_timestamps(ControlMessage& self);

    /**
     * @brief Retrieves timestamps for all keys that match a regex pattern from the ControlMessage object.
     *
     * @param self Reference to the underlying ControlMessage object.
     * @param regex_filter The regex pattern that keys must match to be included in the result.
     * @return A Python dictionary of matching keys and their timestamps as datetime.datetime objects.
     *
     * This method retrieves all timestamps within the ControlMessage object that match a specified
     * regex pattern. Each key and its associated timestamp are returned in a Python dictionary, with
     * timestamps represented as datetime.datetime objects.
     */
    static pybind11::dict filter_timestamp(ControlMessage& self, const std::string& regex_filter);
};

}  // namespace morpheus
