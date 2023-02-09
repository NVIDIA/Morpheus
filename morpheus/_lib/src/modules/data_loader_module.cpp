/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/messages/meta.hpp"

#include <mrc/modules/segment_modules.hpp>
#include <mrc/segment/builder.hpp>
#include <mrc/utils/type_utils.hpp>
#include <nlohmann/json.hpp>

#include <string>

namespace morpheus {
using namespace mrc::modules;

DataLoaderModule::DataLoaderModule(std::string module_name) : SegmentModule(module_name) {}

DataLoaderModule::DataLoaderModule(std::string module_name, nlohmann::json config) :
  SegmentModule(std::move(module_name), std::move(config))
{}

void DataLoaderModule::initialize(mrc::segment::Builder& builder)
{
    if (config().contains("loaders"))
    {
        // TODO
    }

    auto loader_node = builder.make_node<MessageControl, MessageMeta>(
        "input",
        rxcpp::operators::map([this](MessageControl& control_message) { return m_data_loader.load(control_message); }));

    register_input_port("input", loader_node);
    register_output_port("output", loader_node);
}

std::string DataLoaderModule::module_type_name() const
{
    return std::string(::mrc::boost_type_name<type_t>());
}
}  // namespace morpheus