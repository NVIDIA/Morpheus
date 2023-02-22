/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/stages/preallocate.hpp"

#include "morpheus/objects/table_info.hpp"  // for TableInfo

#include <exception>

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

void preallocate(std::shared_ptr<morpheus::MultiMessage> msg,
                 const std::vector<std::tuple<std::string, morpheus::DType>>& columns)
{
    preallocate(msg->meta, columns);
}
//@}
}  // namespace

namespace morpheus {

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
            [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
            [&]() { output.on_completed(); }));
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
