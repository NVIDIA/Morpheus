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

#include "morpheus/stages/serialize.hpp"

#include "mrc/segment/builder.hpp"  // for Builder
#include "mrc/segment/object.hpp"   // for Object

#include "morpheus/messages/meta.hpp"       // for MessageMeta, SlicedMessageMeta
#include "morpheus/objects/table_info.hpp"  // for TableInfo

#include <exception>  // for exception_ptr
#include <memory>     // for shared_ptr, make_shared
#include <string>     // for string
#include <utility>    // for move

// IWYU thinks basic_stringbuf & map are needed for the regex constructor
// IWYU pragma: no_include <map>
// IWYU pragma: no_include <sstream>

namespace morpheus {

constexpr std::regex_constants::syntax_option_type RegexOptions =
    std::regex_constants::ECMAScript | std::regex_constants::icase;

SerializeStage::SerializeStage(const std::vector<std::string>& include,
                               const std::vector<std::string>& exclude,
                               bool fixed_columns) :
  base_t(base_t::op_factory_from_sub_fn(build_operator())),
  m_fixed_columns{fixed_columns}
{
    make_regex_objs(include, m_include);
    make_regex_objs(exclude, m_exclude);
}

void SerializeStage::make_regex_objs(const std::vector<std::string>& regex_strs, std::vector<std::regex>& regex_objs)
{
    for (const auto& s : regex_strs)
    {
        regex_objs.emplace_back(s, RegexOptions);
    }
}

bool SerializeStage::match_column(const std::vector<std::regex>& patterns, const std::string& column) const
{
    for (const auto& re : patterns)
    {
        if (std::regex_match(column, re))
        {
            return true;
        }
    }
    return false;
}

bool SerializeStage::include_column(const std::string& column) const
{
    if (m_include.empty())
    {
        return true;
    }
    else
    {
        return match_column(m_include, column);
    }
}

bool SerializeStage::exclude_column(const std::string& column) const
{
    return match_column(m_exclude, column);
}

std::shared_ptr<SlicedMessageMeta> SerializeStage::get_meta(sink_type_t& msg)
{
    // If none of the columns match the include regex patterns or are all are excluded this has the effect of including
    // all of the rows since calling msg->get_meta({}) will return a view with all columns. The Python impl appears to
    // have the same behavior.
    if (!m_fixed_columns || m_column_names.empty())
    {
        m_column_names.clear();

        std::vector<std::string> column_names;

        column_names = msg->payload()->get_info().get_column_names();

        for (const auto& c : column_names)
        {
            if (include_column(c) && !exclude_column(c))
            {
                m_column_names.push_back(c);
            }
        }
    }

    return std::make_shared<SlicedMessageMeta>(msg->payload(), 0, msg->payload()->count(), m_column_names);
}

SerializeStage::subscribe_fn_t SerializeStage::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output](sink_type_t msg) {
                auto next_meta = this->get_meta(msg);

                output.on_next(std::move(next_meta));
            },
            [&](std::exception_ptr error_ptr) {
                output.on_error(error_ptr);
            },
            [&]() {
                output.on_completed();
            }));
    };
}

// ************ SerializeStageInterfaceProxy ************* //
std::shared_ptr<mrc::segment::Object<SerializeStage>> SerializeStageInterfaceProxy::init(
    mrc::segment::Builder& builder,
    const std::string& name,
    const std::vector<std::string>& include,
    const std::vector<std::string>& exclude,
    bool fixed_columns)
{
    auto stage = builder.construct_object<SerializeStage>(name, include, exclude, fixed_columns);

    return stage;
}

}  // namespace morpheus
