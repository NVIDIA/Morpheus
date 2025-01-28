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
#include "morpheus/io/data_loader.hpp"
#include "morpheus/messages/control.hpp"

#include <nlohmann/json.hpp>

#include <memory>

namespace morpheus {

/**
 * @brief Very simple raw data loader that takes payload data on the control message and returns it
 *
 */
class MORPHEUS_EXPORT PayloadDataLoader : public Loader
{
  public:
    ~PayloadDataLoader() = default;

    PayloadDataLoader() = default;
    PayloadDataLoader(nlohmann::json config);

    std::shared_ptr<ControlMessage> load(std::shared_ptr<ControlMessage> control_message, nlohmann::json task) final;
};

}  // namespace morpheus