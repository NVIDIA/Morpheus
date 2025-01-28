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
#include "morpheus/objects/file_types.hpp"
#include "morpheus/objects/table_info.hpp"
#include "morpheus/utilities/string_util.hpp"

#include <pybind11/pytypes.h>

#include <fstream>  // IWYU pragma: keep
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace morpheus {

/**
 * @addtogroup IO
 * @{
 * @file
 */

/**
 * @brief Serialize a dataframe to an output stream in CSV format
 *
 * @param tbl : A wrapper around data in the dataframe
 * @param out_stream : Output stream to write the results to a destination
 * @param include_header : Determines whether or not to include the header
 * @param include_index_col : Determines whether or not to include the dataframe index
 * @param flush : When `true` flush `out_stream`.
 */
void MORPHEUS_EXPORT df_to_csv(const TableInfo& tbl,
                               std::ostream& out_stream,
                               bool include_header,
                               bool include_index_col = true,
                               bool flush             = false);

/**
 * @brief Serialize a dataframe to an output stream in JSON format
 *
 * @param tbl : A wrapper around data in the dataframe
 * @param include_header : Determines whether or not to include the header
 * @param include_index_col : Determines whether or not to include the dataframe index
 * @return std::string
 */
std::string MORPHEUS_EXPORT df_to_csv(const TableInfo& tbl, bool include_header, bool include_index_col = true);

/**
 * @brief Serialize a dataframe into a JSON formatted string
 * @param tbl : A wrapper around data in the dataframe
 * @param out_stream : Output stream to write the results to a destination
 * @param include_index_col : Determines whether or not to include the dataframe index
 * @param flush : When `true` flush `out_stream`.
 */
void MORPHEUS_EXPORT df_to_json(const TableInfo& tbl,
                                std::ostream& out_stream,
                                bool include_index_col = true,
                                bool flush             = false);

/**
 * @brief Serialize a dataframe into a JSON formatted string
 * @param tbl : A wrapper around data in the dataframe
 * @param include_index_col : Determines whether or not to include the dataframe index
 * @return std::string
 *
 * Note the include_index_col is currently being ignored in both versions of `df_to_json` due to a known issue in
 * Pandas: https://github.com/pandas-dev/pandas/issues/37600
 */
std::string MORPHEUS_EXPORT df_to_json(const TableInfo& tbl, bool include_index_col = true);

/**
 * @brief Serialize a dataframe to an output stream in Parquet format
 *
 * @param tbl : A wrapper around data in the dataframe
 * @param out_stream : Output stream to write the results to a destination
 * @param include_header : Determines whether or not to include the header
 * @param include_index_col : Determines whether or not to include the dataframe index
 * @param flush : When `true` flush `out_stream`.
 */
void MORPHEUS_EXPORT df_to_parquet(const TableInfo& tbl,
                                   std::ostream& out_stream,
                                   bool include_header,
                                   bool include_index_col = true,
                                   bool flush             = false);

/**
 * @brief Serialize a dataframe to an output stream in Parquet format
 *
 * @param tbl : A wrapper around data in the dataframe
 * @param include_header : Determines whether or not to include the header
 * @param include_index_col : Determines whether or not to include the dataframe index
 * @return std::string
 */
std::string MORPHEUS_EXPORT df_to_parquet(const TableInfo& tbl, bool include_header, bool include_index_col = true);

/**
 * @brief Loads a cudf table from a CSV, JSON or Parquet file returning the DataFrame as a Python object
 *
 * @param filename : Name of the file that should be loaded into a table
 * @return pybind11::object
 */
template <typename... ArgsT>
void MORPHEUS_EXPORT write_df_to_file(const TableInfo& tbl,
                                      const std::string& filename,
                                      FileTypes file_type = FileTypes::Auto,
                                      ArgsT&&... args)
{
    if (file_type == FileTypes::Auto)
    {
        file_type = determine_file_type(filename);  // throws if it is unable to determine the type
    }

    std::ofstream out_file;
    out_file.open(filename);

    switch (file_type)
    {
    case FileTypes::JSON: {
        return df_to_json(tbl, out_file, std::forward<ArgsT>(args)...);
        break;
    }
    case FileTypes::CSV: {
        return df_to_csv(tbl, out_file, std::forward<ArgsT>(args)...);
        break;
    }
    case FileTypes::PARQUET: {
        return df_to_parquet(tbl, out_file, std::forward<ArgsT>(args)...);
        break;
    }
    case FileTypes::Auto:
    default:
        throw std::logic_error(MORPHEUS_CONCAT_STR("Unsupported filetype: " << file_type));
    }
}

template <typename... ArgsT>
void MORPHEUS_EXPORT write_df_to_file(const MutableTableInfo& tbl,
                                      const std::string& filename,
                                      FileTypes file_type = FileTypes::Auto,
                                      ArgsT&&... args)
{
    if (file_type == FileTypes::Auto)
    {
        file_type = determine_file_type(filename);  // throws if it is unable to determine the type
    }

    std::ofstream out_file;
    out_file.open(filename);

    switch (file_type)
    {
    case FileTypes::JSON: {
        return df_to_json(tbl, out_file, std::forward<ArgsT>(args)...);
        break;
    }
    case FileTypes::CSV: {
        return df_to_csv(tbl, out_file, std::forward<ArgsT>(args)...);
        break;
    }
    case FileTypes::PARQUET: {
        return df_to_parquet(tbl, out_file, std::forward<ArgsT>(args)...);
        break;
    }
    case FileTypes::Auto:
    default:
        throw std::logic_error(MORPHEUS_CONCAT_STR("Unsupported filetype: " << file_type));
    }
}

struct MORPHEUS_EXPORT SerializersProxy
{
    static void write_df_to_file(pybind11::object df,
                                 std::string filename,
                                 FileTypes file_type,
                                 const pybind11::kwargs& kwargs);
};
/** @} */  // end of group
}  // namespace morpheus
