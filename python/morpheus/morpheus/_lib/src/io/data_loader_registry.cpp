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

#include "morpheus/io/data_loader.hpp"
#include "morpheus/io/loaders/lambda.hpp"
#include "morpheus/messages/control.hpp"
#include "morpheus/objects/factory_registry.hpp"

#include <glog/logging.h>
#include <nlohmann/json.hpp>
#include <pybind11/gil.h>  // for gil_scoped_acquire
#include <pybind11/pybind11.h>
#include <pymrc/utils.hpp>

#include <iostream>
#include <utility>  // for move

namespace morpheus {
template class FactoryRegistry<Loader>;

void LoaderRegistryProxy::register_proxy_factory_fn(
    const std::string& name,
    std::function<std::shared_ptr<ControlMessage>(std::shared_ptr<ControlMessage> control_message, pybind11::dict task)>
        proxy_constructor,
    bool throw_if_exists)
{
    LoaderRegistry::register_factory_fn(
        name,
        [proxy_constructor](nlohmann::json config) {
            return std::make_shared<LambdaLoader>(
                [proxy_constructor](std::shared_ptr<ControlMessage> control_message, nlohmann::json task) {
                    pybind11::gil_scoped_acquire gil;
                    auto py_task = mrc::pymrc::cast_from_json(task);
                    return std::move(proxy_constructor(control_message, py_task));
                },
                config);
        },
        throw_if_exists);

    register_factory_cleanup_fn(name);
}

void LoaderRegistryProxy::register_factory_cleanup_fn(const std::string& name)
{
    {
        auto at_exit = pybind11::module_::import("atexit");
        at_exit.attr("register")(pybind11::cpp_function([name]() {
            VLOG(2) << "(atexit) Unregistering loader: " << name;

            // Try unregister -- ignore if already unregistered
            FactoryRegistry<Loader>::unregister_factory_fn(name, false);
        }));
    }
}

}  // namespace morpheus
