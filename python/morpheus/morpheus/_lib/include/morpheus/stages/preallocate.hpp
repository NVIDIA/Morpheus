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
#include "morpheus/messages/control.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/objects/dtype.hpp"  // for TypeId

#include <mrc/node/sink_properties.hpp>
#include <mrc/node/source_properties.hpp>
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>
#include <pymrc/node.hpp>
#include <rxcpp/rx.hpp>

#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace morpheus {

namespace {
/**
 * @brief Performs preallocation to the underlying dataframe. These functions ensure that the MutableTableInfo object
 * has gone out of scope and thus releasing the mutex prior to the stage calling `on_next` which may block.
 *
 * @param msg
 * @param columns
 */
//@{
void preallocate(std::shared_ptr<morpheus::MessageMeta> msg,
                 const std::vector<std::tuple<std::string, morpheus::DType>>& columns)
{
    auto table = msg->get_mutable_info();
    table.insert_missing_columns(columns);
}

void preallocate(std::shared_ptr<morpheus::ControlMessage> msg,
                 const std::vector<std::tuple<std::string, morpheus::DType>>& columns)
{
    preallocate(msg->payload(), columns);
}

}  // namespace

/****** Component public implementations *******************/
/****** PreallocateStage ********************************/
/* Preallocates new columns into the underlying dataframe. This stage supports both MessageMeta & ControlMessage. In the
 * Python bindings the stage is bound as `PreallocateMessageMetaStage` and `PreallocateControlMessageStage`
 */
template <typename MessageT>
class MORPHEUS_EXPORT PreallocateStage
  : public mrc::pymrc::PythonNode<std::shared_ptr<MessageT>, std::shared_ptr<MessageT>>
{
  public:
    using base_t = mrc::pymrc::PythonNode<std::shared_ptr<MessageT>, std::shared_ptr<MessageT>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    PreallocateStage(const std::vector<std::tuple<std::string, TypeId>>& needed_columns);

  private:
    subscribe_fn_t build_operator();

    std::vector<std::tuple<std::string, DType>> m_needed_columns;
};

/****** DeserializationStageInterfaceProxy******************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
template <typename MessageT>
struct MORPHEUS_EXPORT PreallocateStageInterfaceProxy
{
    /**
     * @brief Create and initialize a DeserializationStage, and return the result.
     */
    static std::shared_ptr<mrc::segment::Object<PreallocateStage<MessageT>>> init(
        mrc::segment::Builder& builder,
        const std::string& name,
        std::vector<std::tuple<std::string, TypeId>> needed_columns);
};

template <typename MessageT>
PreallocateStage<MessageT>::PreallocateStage(const std::vector<std::tuple<std::string, TypeId>>& needed_columns) :
  base_t(base_t::op_factory_from_sub_fn(build_operator()))
{
    for (const auto& col : needed_columns)
    {
        m_needed_columns.emplace_back(std::make_tuple<>(std::get<0>(col), DType(std::get<1>(col))));
    }
}

template <typename MessageT>
typename PreallocateStage<MessageT>::subscribe_fn_t PreallocateStage<MessageT>::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output](sink_type_t x) {
                // Since the msg was just emitted from the source we shouldn't have any trouble acquiring the mutex.
                preallocate(x, m_needed_columns);
                output.on_next(std::move(x));
            },
            [&](std::exception_ptr error_ptr) {
                output.on_error(error_ptr);
            },
            [&]() {
                output.on_completed();
            }));
    };
}

template <typename MessageT>
std::shared_ptr<mrc::segment::Object<PreallocateStage<MessageT>>> PreallocateStageInterfaceProxy<MessageT>::init(
    mrc::segment::Builder& builder,
    const std::string& name,
    std::vector<std::tuple<std::string, TypeId>> needed_columns)
{
    return builder.construct_object<PreallocateStage<MessageT>>(name, needed_columns);
}

}  // namespace morpheus
