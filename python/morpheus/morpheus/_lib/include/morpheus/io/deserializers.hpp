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
#include "morpheus/objects/file_types.hpp"  // for FileTypes

#include <cudf/io/types.hpp>
#include <pybind11/pytypes.h>  // for pybind11::object

#include <optional>
#include <string>
#include <vector>

namespace morpheus {

/**
 * @addtogroup IO
 * @{
 * @file
 */

/**
 * @brief Get the column names from table object. Looks at both column_names as well as schema_info and returns the
 * correct one.
 *
 * @param table The table to pull the columns from
 * @return std::vector<std::string>
 */
std::vector<std::string> MORPHEUS_EXPORT get_column_names_from_table(const cudf::io::table_with_metadata& table);

/**
 * @brief Loads a cudf table from either CSV or JSON file
 *
 * @param filename : Name of the file that should be loaded into a table
 * @return cudf::io::table_with_metadata
 */
cudf::io::table_with_metadata MORPHEUS_EXPORT load_table_from_file(const std::string& filename,
                                                                   FileTypes file_type            = FileTypes::Auto,
                                                                   std::optional<bool> json_lines = std::nullopt);

/**
 * @brief Returns the number of index columns in `data_table`, in practice this will be a `0` or `1`
 *
 * @param data_table : Table which contains the data and it's metadata
 * @return int
 */
int MORPHEUS_EXPORT get_index_col_count(const cudf::io::table_with_metadata& data_table);

/**
 * @brief Returns the number of index columns in `data_table`, in practice this will be a `0` or `1`
 * If `data_table` contains a column named "Unnamed: 0" it will be renamed to ""
 *
 * @param data_table : Table which contains the data and it's metadata
 * @return int
 */
int MORPHEUS_EXPORT prepare_df_index(cudf::io::table_with_metadata& data_table);

/**
 * @brief Loads a cudf table from either CSV, JSON or Parquet file returning the DataFrame as a Python object
 *
 * @param filename : Name of the file that should be loaded into a table
 * @return pybind11::object
 */
pybind11::object MORPHEUS_EXPORT read_file_to_df(const std::string& filename, FileTypes file_type = FileTypes::Auto);

/** @} */  // end of group
}  // namespace morpheus
