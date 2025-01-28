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

#include "morpheus/export.h"
#include "morpheus/io/data_loader.hpp"

#include <glog/logging.h>
#include <mrc/utils/type_utils.hpp>
#include <pybind11/pybind11.h>

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace morpheus {

template <typename ObjectReturnTypeT>
class MORPHEUS_EXPORT FactoryRegistry
{
  public:
    static bool contains(const std::string& name)
    {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        return m_object_constructors.count(name) > 0;
    }

    static std::vector<std::string> list()
    {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        std::vector<std::string> names;
        for (const auto& [name, _] : m_object_constructors)
        {
            names.push_back(name);
        }
        return names;
    }

    static std::shared_ptr<ObjectReturnTypeT> create_object_from_factory(const std::string& name,
                                                                         nlohmann::json config = {})
    {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        VLOG(2) << "Retrieving factory constructor: " << name << "(" << mrc::type_name<ObjectReturnTypeT>() << ")";

        if (m_object_constructors.count(name) == 0)
        {
            throw std::runtime_error("Unknown data loader: " + name);
        }

        return m_object_constructors[name](config);
    }

    static void register_factory_fn(const std::string& name,
                                    const std::function<std::shared_ptr<ObjectReturnTypeT>(nlohmann::json)>& loader_fn,
                                    bool throw_if_exists = true)
    {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        VLOG(2) << "Registering factory constructor: " << name << "(" << mrc::type_name<ObjectReturnTypeT>() << ")";
        if (m_object_constructors.count(name) > 0)
        {
            if (throw_if_exists)
            {
                throw std::runtime_error("Duplicate data loader registration: " + name);
            }
        }
        m_object_constructors[name] = loader_fn;
    }

    static void unregister_factory_fn(const std::string& name, bool throw_if_missing = true)
    {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        VLOG(2) << "Un-registering factory constructor: " << name << "(" << mrc::type_name<ObjectReturnTypeT>() << ")";
        if (m_object_constructors.count(name) == 0)
        {
            if (throw_if_missing)
            {
                throw std::runtime_error("Unknown data loader: " + name);
            }

            return;
        }
        m_object_constructors.erase(name);
    }

  private:
    static std::mutex m_mutex;
    static std::map<std::string, std::function<std::shared_ptr<ObjectReturnTypeT>(nlohmann::json)>>
        m_object_constructors;
};

template <typename ObjectReturnTypeT>
std::mutex FactoryRegistry<ObjectReturnTypeT>::m_mutex;

template <typename ObjectReturnTypeT>
std::map<std::string, std::function<std::shared_ptr<ObjectReturnTypeT>(nlohmann::json)>>
    FactoryRegistry<ObjectReturnTypeT>::m_object_constructors;

}  // namespace morpheus
