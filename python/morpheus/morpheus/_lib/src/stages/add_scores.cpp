/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "morpheus/stages/add_scores.hpp"

#include "mrc/segment/builder.hpp"  // for Builder
#include "mrc/segment/object.hpp"   // for Object

#include "morpheus/stages/add_scores_stage_base.hpp"  // for AddScoresStageBase

#include <cstddef>   // for size_t
#include <map>       // for map
#include <memory>    // for shared_ptr
#include <optional>  // for nullopt, optional
#include <utility>   // for move

namespace morpheus {

// Component public implementations
// ************ AddScoresStage **************************** //
AddScoresStage::AddScoresStage(std::map<std::size_t, std::string> idx2label) :
  AddScoresStageBase(std::move(idx2label), std::nullopt)
{}

// ************ AddScoresStageInterfaceProxy ************* //
std::shared_ptr<mrc::segment::Object<AddScoresStage>> AddScoresStageInterfaceProxy::init(
    mrc::segment::Builder& builder, const std::string& name, std::map<std::size_t, std::string> idx2label)
{
    return builder.construct_object<AddScoresStage>(name, std::move(idx2label));
}

}  // namespace morpheus
