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

#include "morpheus/utilities/table_util.hpp"

#include <cudf/io/csv.hpp>
#include <cudf/io/json.hpp>
#include <cudf/stream_compaction.hpp>  // for drop_nulls
#include <cudf/types.hpp>              // for size_type
#include <glog/logging.h>
#include <pybind11/pybind11.h>

#include <algorithm>  // for find, transform
#include <filesystem>
#include <iterator>  // for back_insert_iterator, back_inserter
#include <memory>    // for unique_ptr
#include <ostream>    // needed for logging
#include <stdexcept>  // for runtime_error

namespace {
namespace fs = std::filesystem;
namespace py = pybind11;
}  // namespace
namespace morpheus {
cudf::io::table_with_metadata CuDFTableUtil::load_table(const std::string& filename)
{
    auto file_path = fs::path(filename);

    if (file_path.extension() == ".json" || file_path.extension() == ".jsonlines")
    {
        // First, load the file into json
        auto options = cudf::io::json_reader_options::builder(cudf::io::source_info{filename}).lines(true);

        return cudf::io::read_json(options.build());
    }
    else if (file_path.extension() == ".csv")
    {
        auto options = cudf::io::csv_reader_options::builder(cudf::io::source_info{filename});

        return cudf::io::read_csv(options.build());
    }
    else
    {
        LOG(FATAL) << "Unknown extension for file: " << filename;
        throw std::runtime_error("Unknown extension");
    }
}

std::vector<std::string> CuDFTableUtil::get_column_names(const cudf::io::table_with_metadata& table)
{
    auto const& schema = table.metadata.schema_info;

    std::vector<std::string> names;
    names.reserve(schema.size());
    std::transform(schema.cbegin(), schema.cend(), std::back_inserter(names), [](auto const& c) {
        return c.name;
    });

    return names;
}

void CuDFTableUtil::filter_null_data(cudf::io::table_with_metadata& table,
                                     const std::vector<std::string>& filter_columns)
{
    std::vector<cudf::size_type> filter_keys;
    auto column_names = get_column_names(table);
    for (const auto& column_name : filter_columns)
    {
        auto found_col = std::find(column_names.cbegin(), column_names.cend(), column_name);
        if (found_col != column_names.cend())
        {
            filter_keys.push_back((found_col - column_names.cbegin()));
        }
    }

    auto tv             = table.tbl->view();
    auto filtered_table = cudf::drop_nulls(tv, filter_keys, filter_keys.size());

    table.tbl.swap(filtered_table);
}
}  // namespace morpheus
