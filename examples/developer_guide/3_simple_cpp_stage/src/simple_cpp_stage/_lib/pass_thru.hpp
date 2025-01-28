/**
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

#include <morpheus/export.h>              // for exporting symbols
#include <morpheus/messages/control.hpp>  // for ControlMessage
#include <mrc/segment/builder.hpp>        // for Segment Builder
#include <mrc/segment/object.hpp>         // for Segment Object
#include <pymrc/node.hpp>                 // for PythonNode
#include <rxcpp/rx.hpp>

#include <memory>
#include <string>
#include <thread>

// IWYU pragma: no_include "morpheus/objects/data_table.hpp"
// IWYU pragma: no_include <boost/fiber/context.hpp>

namespace morpheus_example {

using namespace morpheus;

// pybind11 sets visibility to hidden by default; we want to export our symbols
class MORPHEUS_EXPORT PassThruStage
  : public mrc::pymrc::PythonNode<std::shared_ptr<ControlMessage>, std::shared_ptr<ControlMessage>>
{
  public:
    using base_t = mrc::pymrc::PythonNode<std::shared_ptr<ControlMessage>, std::shared_ptr<ControlMessage>>;
    using base_t::sink_type_t;
    using base_t::source_type_t;
    using base_t::subscribe_fn_t;

    PassThruStage();

    subscribe_fn_t build_operator();
};

struct MORPHEUS_EXPORT PassThruStageInterfaceProxy
{
    static std::shared_ptr<mrc::segment::Object<PassThruStage>> init(mrc::segment::Builder& builder,
                                                                     const std::string& name);
};

}  // namespace morpheus_example
