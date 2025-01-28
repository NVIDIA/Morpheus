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

#pragma once

#include "morpheus/export.h"
#include "morpheus/messages/control.hpp"
#include "morpheus/messages/meta.hpp"

#include <nlohmann/json.hpp>

#include <map>
#include <memory>
#include <string>

namespace morpheus {

/**
 * @brief Abstract class for loading data from a source.
 *
 * This class defines two virtual methods for loading and returning control messages, as well as a
 * protected method for accessing the configuration information. It can be extended to implement
 * specific loading logic for different types of data sources.
 */
class MORPHEUS_EXPORT Loader
{
  public:
    /**
     * @brief Destructor for the Loader class.
     */
    ~Loader() = default;

    /**
     * @brief Default constructor for the Loader class.
     */
    Loader() = default;

    /**
     * @brief Constructor for the Loader class.
     * @param config A json object containing configuration information for the loader.
     */
    Loader(nlohmann::json config);

    /**
     * @brief Virtual method for extracting message payload from a control message.
     * @param message A shared pointer to the control message containing the payload.
     * @return A shared pointer to the message payload as a MessageMeta object.
     */
    virtual std::shared_ptr<MessageMeta> payload(std::shared_ptr<ControlMessage> message);

    /**
     * @brief Virtual method for loading a control message.
     * @param message A shared pointer to the control message to be loaded.
     * @param task A json object describing the loading task.
     * @return A shared pointer to the loaded control message.
     */
    virtual std::shared_ptr<ControlMessage> load(std::shared_ptr<ControlMessage> message, nlohmann::json task);

  protected:
    /**
     * @brief Protected method for accessing the loader's configuration information.
     * @return A json object containing configuration information for the loader.
     */
    nlohmann::json config() const;

  private:
    nlohmann::json m_config{};  // Configuration information for the loader.
};

/**
 * @brief Class for managing and loading data using different loaders.
 *
 * This class manages a set of Loader objects and provides methods for registering, removing, and
 * loading data using these objects. It also defines a method for loading control messages from data
 * sources using the registered loaders.
 */
class MORPHEUS_EXPORT DataLoader
{
  public:
    /**
     * @brief Default constructor for the DataLoader class.
     */
    DataLoader() = default;

    /**
     * @brief Constructor with config for the DataLoader class.
     */
    DataLoader(nlohmann::json config);

    /**
     * @brief Destructor for the DataLoader class.
     */
    ~DataLoader() = default;

    /**
     * @brief Method for loading a control message using the registered loaders.
     * @param control_message A shared pointer to the control message to be loaded.
     * @return A shared pointer to the loaded control message.
     */
    std::shared_ptr<ControlMessage> load(std::shared_ptr<ControlMessage> control_message);

    /**
     * @brief Method for registering a loader instance with the data loader.
     * @param loader_id A string identifier for the loader instance.
     * @param loader A shared pointer to the Loader object to be registered.
     * @param overwrite A boolean indicating whether to overwrite an existing loader instance with
     *                  the same identifier.
     */
    void add_loader(const std::string& loader_id, std::shared_ptr<Loader> loader, bool overwrite = true);

    /**
     * @brief Method for removing a loader instance from the data loader.
     * @param loader_id A string identifier for the loader instance to be removed.
     * @param throw_if_not_found A boolean indicating whether to throw an exception if the loader
     *                           instance with the given identifier is not found.
     */
    void remove_loader(const std::string& loader_id, bool throw_if_not_found = true);

  private:
    std::map<std::string, std::shared_ptr<Loader>> m_loaders{};  // Map of registered loader instances.
    nlohmann::json m_config;
};

}  // namespace morpheus