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

#include "morpheus/export.h"                  // for MORPHEUS_EXPORT
#include "morpheus/messages/control.hpp"      // for ControlMessage
#include "morpheus/messages/meta.hpp"         // for MessageMeta
#include "morpheus/types.hpp"                 // for TensorIndex
#include "morpheus/utilities/json_types.hpp"  // for control_message_task_t

#include <boost/fiber/context.hpp>  // for operator<<
#include <mrc/segment/builder.hpp>  // for Builder
#include <mrc/segment/object.hpp>   // for Object
#include <nlohmann/json.hpp>        // for basic_json, json
#include <pybind11/pytypes.h>       // for object
#include <pymrc/node.hpp>           // for PythonNode
#include <rxcpp/rx.hpp>             // for decay_t, trace_activity, from, observable_member

#include <memory>   // for shared_ptr, unique_ptr
#include <string>   // for string
#include <thread>   // for operator<<
#include <utility>  // for move, pair

namespace morpheus {
/****** Component public implementations *******************/

/**
 * @addtogroup stages
 * @{
 * @file
 */

/****** DeserializationStage********************************/
class MORPHEUS_EXPORT DeserializeStage
  : public mrc::pymrc::PythonNode<std::shared_ptr<MessageMeta>, std::shared_ptr<ControlMessage>>
{
  public:
    using base_t = mrc::pymrc::PythonNode<std::shared_ptr<MessageMeta>, std::shared_ptr<ControlMessage>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    /**
     * @brief Construct a new Deserialize Stage object
     *
     * @param batch_size Number of messages to be divided into each batch
     * @param ensure_sliceable_index Whether or not to call
     * `ensure_sliceable_index()` on all incoming `MessageMeta`
     * @param task Optional task to be added to all outgoing `ControlMessage`s
     */
    DeserializeStage(TensorIndex batch_size,
                     bool ensure_sliceable_index                  = true,
                     std::unique_ptr<control_message_task_t> task = nullptr) :
      base_t(base_t::op_factory_from_sub_fn(build_operator())),
      m_batch_size(batch_size),
      m_ensure_sliceable_index(ensure_sliceable_index),
      m_task(std::move(task)){};

  private:
    subscribe_fn_t build_operator();

    TensorIndex m_batch_size;
    bool m_ensure_sliceable_index{true};
    std::unique_ptr<control_message_task_t> m_task{nullptr};
};

/****** DeserializationStageInterfaceProxy******************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MORPHEUS_EXPORT DeserializeStageInterfaceProxy
{
    /**
     * @brief Create and initialize a DeserializationStage that emits
     * ControlMessage's, and return the result. If `task_type` is not None,
     * `task_payload` must also be not None, and vice versa.
     *
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @param batch_size : Number of messages to be divided into each batch
     * @param ensure_sliceable_index Whether or not to call
     * `ensure_sliceable_index()` on all incoming `MessageMeta`
     * @param task_type : Optional task type to be added to all outgoing messages
     * @param task_payload : Optional json object describing the task to be added
     * to all outgoing messages
     * @return std::shared_ptr<mrc::segment::Object<DeserializeStage>>
     */
    static std::shared_ptr<mrc::segment::Object<DeserializeStage>> init(mrc::segment::Builder& builder,
                                                                        const std::string& name,
                                                                        TensorIndex batch_size,
                                                                        bool ensure_sliceable_index,
                                                                        const pybind11::object& task_type,
                                                                        const pybind11::object& task_payload);
};

/** @} */  // end of group
}  // namespace morpheus
