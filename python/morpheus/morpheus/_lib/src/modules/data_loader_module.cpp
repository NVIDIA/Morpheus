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

#include "morpheus/modules/data_loader_module.hpp"

#include "morpheus/io/data_loader_registry.hpp"
#include "morpheus/messages/control.hpp"

#include <glog/logging.h>
#include <mrc/modules/segment_modules.hpp>
#include <mrc/node/rx_node.hpp>
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>
#include <mrc/utils/type_utils.hpp>
#include <nlohmann/json.hpp>
#include <rxcpp/rx.hpp>
// IWYU pragma: no_include "rxcpp/sources/rx-iterate.hpp"

#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>

using namespace mrc::modules;
using nlohmann::json;

namespace morpheus {

const std::string DataLoaderModule::s_config_schema = R"()";

DataLoaderModule::~DataLoaderModule()
{
    VLOG(30) << "DataLoaderModule::~DataLoaderModule(): " << name() << std::endl;
}

DataLoaderModule::DataLoaderModule(std::string module_name) : SegmentModule(module_name) {}

DataLoaderModule::DataLoaderModule(std::string module_name, nlohmann::json _config) :
  SegmentModule(std::move(module_name), std::move(_config))
{
    if (config().contains("loader_config"))
    {
        auto loader_config = config()["loader_config"];
        m_data_loader      = DataLoader(loader_config);
    }

    if (config().contains("loaders"))
    {
        // TODO(Devin): Add schema validation
    }

    if (config().contains("loaders") and config()["loaders"].is_array() and !config()["loaders"].empty())
    {
        auto loader_list = config()["loaders"];
        for (json::iterator it = loader_list.begin(); it != loader_list.end(); ++it)
        {
            auto loader_id_it = it.value().find("id");
            if (loader_id_it == it.value().end())
            {
                throw std::runtime_error("Loader id not specified");
            }

            auto loader_id         = loader_id_it.value().get<std::string>();
            auto loader_properties = it->value("properties", json({}));
            if (LoaderRegistry::contains(loader_id))
            {
                VLOG(2) << "Adding loader: " << loader_id << " with properties: " << loader_properties.dump(2);
                m_data_loader.add_loader(loader_id,
                                         LoaderRegistry::create_object_from_factory(loader_id, loader_properties));
            }
            else
            {
                throw std::runtime_error("Unknown or unsupported loader type: " + loader_id);
            }
        }
    }
    else
    {
        LOG(WARNING) << "No loaders specified in config: " << config().dump(2);
    }
}

void DataLoaderModule::initialize(mrc::segment::IBuilder& builder)
{
    auto loader_node = builder.make_node<std::shared_ptr<ControlMessage>, std::shared_ptr<ControlMessage>>(
        "input", rxcpp::operators::map([this](std::shared_ptr<ControlMessage> control_message) {
            return m_data_loader.load(control_message);
        }));

    builder.register_module_input("input", loader_node);
    builder.register_module_output("output", loader_node);
}

std::string DataLoaderModule::module_type_name() const
{
    return std::string(::mrc::type_name<type_t>());
}
}  // namespace morpheus
