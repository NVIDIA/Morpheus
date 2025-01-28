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

#include "morpheus/stages/file_source.hpp"

#include "morpheus/io/deserializers.hpp"
#include "morpheus/objects/file_types.hpp"
#include "morpheus/objects/table_info.hpp"
#include "morpheus/utilities/cudf_util.hpp"
#include "morpheus/utilities/table_util.hpp"  // for filter_null_data

#include <cudf/types.hpp>
#include <glog/logging.h>
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>
#include <pybind11/cast.h>  // IWYU pragma: keep
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>  // for str_attr_accessor
#include <pybind11/pytypes.h>   // for pybind11::int_
#include <pymrc/node.hpp>

#include <filesystem>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>  // for invalid_argument
#include <utility>

namespace morpheus {
// Component public implementations
// ************ FileSourceStage ************* //
FileSourceStage::FileSourceStage(std::string filename,
                                 int repeat,
                                 bool filter_null,
                                 std::vector<std::string> filter_null_columns,
                                 std::optional<bool> json_lines) :
  PythonSource(build()),
  m_filename(std::move(filename)),
  m_repeat(repeat),
  m_filter_null(filter_null),
  m_filter_null_columns(std::move(filter_null_columns)),
  m_json_lines(json_lines)
{
    if (m_filter_null && m_filter_null_columns.empty())
    {
        throw std::invalid_argument("Filter null columns must not be empty if filter_null is true");
    }
}

FileSourceStage::subscriber_fn_t FileSourceStage::build()
{
    return [this](rxcpp::subscriber<source_type_t> output) {
        auto data_table = load_table_from_file(m_filename, FileTypes::Auto, m_json_lines);
        if (m_filter_null)
        {
            CuDFTableUtil::filter_null_data(data_table, m_filter_null_columns);
        }

        int index_col_count = prepare_df_index(data_table);

        // Next, create the message metadata. This gets reused for repeats
        // When index_col_count is 0 this will cause a new range index to be created
        auto meta = MessageMeta::create_from_cpp(std::move(data_table), index_col_count);

        // next_meta stores a copy of the upcoming meta
        std::shared_ptr<MessageMeta> next_meta = nullptr;

        for (cudf::size_type repeat_idx = 0; repeat_idx < m_repeat; ++repeat_idx)
        {
            if (!output.is_subscribed())
            {
                // Grab the GIL before disposing, just in case
                pybind11::gil_scoped_acquire gil;

                // Reset meta to allow the DCHECK after the loop to pass
                meta.reset();

                break;
            }

            // Clone the meta object before pushing while we still have access to it
            if (repeat_idx + 1 < m_repeat)
            {
                // Use the copy function, copy_to_py_object will acquire it's own gil
                auto df = CudfHelper::table_from_table_info(meta->get_info());

                // GIL must come after get_info
                pybind11::gil_scoped_acquire gil;

                pybind11::int_ df_len = pybind11::len(df);

                pybind11::object index = df.attr("index");

                df.attr("index") = index + df_len;

                next_meta = MessageMeta::create_from_python(std::move(df));
            }

            DCHECK(meta) << "Cannot push null meta";

            output.on_next(std::move(meta));

            // Move next_meta into meta
            std::swap(meta, next_meta);
        }

        DCHECK(!meta) << "meta was not properly pushed";
        DCHECK(!next_meta) << "next_meta was not properly pushed";

        output.on_completed();
    };
}

// ************ FileSourceStageInterfaceProxy ************ //
std::shared_ptr<mrc::segment::Object<FileSourceStage>> FileSourceStageInterfaceProxy::init(
    mrc::segment::Builder& builder,
    const std::string& name,
    std::string filename,
    int repeat,
    bool filter_null,
    std::vector<std::string> filter_null_columns,
    pybind11::dict parser_kwargs)
{
    std::optional<bool> json_lines = std::nullopt;

    if (parser_kwargs.contains("lines"))
    {
        json_lines = parser_kwargs["lines"].cast<bool>();
    }

    auto stage = builder.construct_object<FileSourceStage>(
        name, filename, repeat, filter_null, std::move(filter_null_columns), json_lines);

    return stage;
}

std::shared_ptr<mrc::segment::Object<FileSourceStage>> FileSourceStageInterfaceProxy::init(
    mrc::segment::Builder& builder,
    const std::string& name,
    std::filesystem::path filename,
    int repeat,
    bool filter_null,
    std::vector<std::string> filter_null_columns,
    pybind11::dict parser_kwargs)
{
    return init(builder,
                name,
                filename.string(),
                repeat,
                filter_null,
                std::move(filter_null_columns),
                std::move(parser_kwargs));
}
}  // namespace morpheus
