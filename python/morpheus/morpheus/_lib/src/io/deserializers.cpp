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

#include "morpheus/io/deserializers.hpp"

#include "morpheus/utilities/cudf_util.hpp"  // for CudfHelper
#include "morpheus/utilities/stage_util.hpp"
#include "morpheus/utilities/string_util.hpp"
#include "morpheus/utilities/table_util.hpp"  // for get_column_names

#include <cudf/column/column.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/json.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>  // IWYU pragma: keep
#include <cudf/types.hpp>        // for cudf::type_id
#include <pybind11/pybind11.h>   // IWYU pragma: keep

#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <utility>
// We're already including pybind11.h, and including only gil.h as IWYU suggests yields an undefined symbol error
// IWYU pragma: no_include <pybind11/gil.h>

namespace {
const std::regex INDEX_REGEX(R"(^\s*(unnamed: 0|id)\s*$)",
                             std::regex_constants::ECMAScript | std::regex_constants::icase);

const std::regex UNNAMED_REGEX(R"(^\s*unnamed: 0\s*$)", std::regex_constants::ECMAScript | std::regex_constants::icase);
}  // namespace

namespace morpheus {

std::vector<std::string> get_column_names_from_table(const cudf::io::table_with_metadata& table)
{
    return foreach_map(table.metadata.schema_info, [](auto schema) {
        return schema.name;
    });
}

cudf::io::table_with_metadata load_table_from_file(const std::string& filename,
                                                   FileTypes file_type,
                                                   std::optional<bool> json_lines)
{
    if (file_type == FileTypes::Auto)
    {
        file_type = determine_file_type(filename);  // throws if it is unable to determine the type
    }

    cudf::io::table_with_metadata table;

    switch (file_type)
    {
    case FileTypes::JSON: {
        auto options =
            cudf::io::json_reader_options::builder(cudf::io::source_info{filename}).lines(json_lines.value_or(true));
        table = cudf::io::read_json(options.build());
        break;
    }
    case FileTypes::CSV: {
        auto options = cudf::io::csv_reader_options::builder(cudf::io::source_info{filename});
        table        = cudf::io::read_csv(options.build());
        break;
    }
    case FileTypes::PARQUET: {
        auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filename});
        table        = cudf::io::read_parquet(options.build());
        break;
    }
    case FileTypes::Auto:
    default:
        throw std::logic_error(MORPHEUS_CONCAT_STR("Unsupported filetype: " << file_type));
    }

    if (!table.tbl)
    {
        throw std::runtime_error(MORPHEUS_CONCAT_STR("Failed to load file '" << filename << "' as type " << file_type));
    }

    return table;
}

pybind11::object read_file_to_df(const std::string& filename, FileTypes file_type)
{
    auto table          = load_table_from_file(filename, file_type);
    int index_col_count = prepare_df_index(table);

    pybind11::gil_scoped_acquire gil;
    return CudfHelper::table_from_table_with_metadata(std::move(table), index_col_count);
}

int get_index_col_count(const cudf::io::table_with_metadata& data_table)
{
    int index_col_count = 0;

    std::vector<std::string> names = CuDFTableUtil::get_column_names(data_table);

    // Check if we have a first column with INT64 data type
    if (names.size() >= 1 && data_table.tbl->get_column(0).type().id() == cudf::type_id::INT64)
    {
        // Get the column name
        const auto& col_name = names[0];

        // Check it against some common terms
        if (std::regex_search(col_name, INDEX_REGEX))
        {
            index_col_count = 1;
        }
    }

    return index_col_count;
}

int prepare_df_index(cudf::io::table_with_metadata& data_table)
{
    const int index_col_count = get_index_col_count(data_table);

    if (index_col_count > 0)
    {
        auto& col_name = data_table.metadata.schema_info[0].name;

        // Also, if its the hideous 'Unnamed: 0', then just use an empty string
        if (std::regex_search(col_name, UNNAMED_REGEX))
        {
            col_name.clear();
        }
    }

    return index_col_count;
}

}  // namespace morpheus
