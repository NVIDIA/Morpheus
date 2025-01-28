/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/export.h"              // for MORPHEUS_EXPORT
#include "morpheus/messages/control.hpp"  // for ControlMessage

#include <boost/fiber/context.hpp>  // for operator<<
#include <pymrc/node.hpp>           // for PythonNode
#include <rxcpp/rx.hpp>             // for observable_member, trace_activity, decay_t, from

#include <cstddef>   // for size_t
#include <map>       // for map
#include <memory>    // for shared_ptr
#include <optional>  // for optional
#include <string>    // for string
#include <thread>    // for operator<<

namespace morpheus {
/****** Component public implementations *******************/
/**************AddScoresStageBase***************************/

/**
 * @addtogroup stages
 * @{
 * @file
 */

/**
 * @brief Base class for both `AddScoresStage` and `AddClassificationStage`
 */
class MORPHEUS_EXPORT AddScoresStageBase
  : public mrc::pymrc::PythonNode<std::shared_ptr<ControlMessage>, std::shared_ptr<ControlMessage>>
{
  public:
    using base_t = mrc::pymrc::PythonNode<std::shared_ptr<ControlMessage>, std::shared_ptr<ControlMessage>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    /**
     * @brief Construct a new Add Classifications Stage object
     *
     * @param threshold : Threshold to consider true/false for each class
     * @param idx2label : Index to classification labels map
     */
    AddScoresStageBase(std::map<std::size_t, std::string> idx2label, std::optional<float> threshold);

    /**
     * Called every time a message is passed to this stage
     */
    source_type_t on_data(sink_type_t msg);

  private:
    std::map<std::size_t, std::string> m_idx2label;
    std::optional<float> m_threshold;

    // The minimum number of columns needed to extract the label data
    std::size_t m_min_col_count;
};

/** @} */  // end of group
}  // namespace morpheus
