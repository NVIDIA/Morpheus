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
#include "morpheus/messages/control.hpp"
#include "morpheus/objects/factory_registry.hpp"

#include <pybind11/pytypes.h>

#include <functional>
#include <memory>
#include <string>

namespace morpheus {

extern template class MORPHEUS_EXPORT FactoryRegistry<Loader>;

using LoaderRegistry = FactoryRegistry<Loader>;  // NOLINT

struct MORPHEUS_EXPORT LoaderRegistryProxy
{
    static void register_proxy_factory_fn(
        const std::string& name,
        std::function<std::shared_ptr<ControlMessage>(std::shared_ptr<ControlMessage>, pybind11::dict)>
            proxy_constructor,
        bool throw_if_exists = true);

    static void register_factory_cleanup_fn(const std::string& name);
};

}  // namespace morpheus
