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

#include "morpheus/export.h"                          // for MORPHEUS_EXPORT
#include "morpheus/stages/add_scores_stage_base.hpp"  // for AddScoresStageBase

#include <mrc/segment/builder.hpp>  // for Builder
#include <mrc/segment/object.hpp>   // for Object

#include <cstddef>  // for size_t
#include <map>      // for map
#include <memory>   // for shared_ptr
#include <string>   // for string
namespace morpheus {

/****** Component public implementations *******************/
/****** AddClassificationStage********************************/

/**
 * @addtogroup stages
 * @{
 * @file
 */

/**
 * @brief Add detected classifications to each message. Classification labels based on probabilities calculated in
 * inference stage. Label indexes will be looked up in the idx2label property.
 */
class MORPHEUS_EXPORT AddClassificationsStage : public AddScoresStageBase
{
  public:
    /**
     * @brief Construct a new Add Classifications Stage object
     *
     * @param threshold : Threshold to consider true/false for each class
     * @param idx2label : Index to classification labels map
     */
    AddClassificationsStage(std::map<std::size_t, std::string> idx2label, float threshold);
};

/****** AddClassificationStageInterfaceProxy******************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MORPHEUS_EXPORT AddClassificationStageInterfaceProxy
{
    /**
     * @brief Create and initialize a AddClassificationStage that receives
     * ControlMessage and emits ControlMessage, and return the result
     *
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @param idx2label : Index to classification labels map
     * @param threshold : Threshold to consider true/false for each class
     * @return std::shared_ptr<mrc::segment::Object<AddClassificationsStage>>
     */
    static std::shared_ptr<mrc::segment::Object<AddClassificationsStage>> init(
        mrc::segment::Builder& builder,
        const std::string& name,
        std::map<std::size_t, std::string> idx2label,
        float threshold);
};
/** @} */  // end of group
}  // namespace morpheus
