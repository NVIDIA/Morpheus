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

#include "morpheus/export.h"  // for MORPHEUS_EXPORT

#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>  // IWYU pragma: keep

#include <string>
#include <vector>

#pragma once

namespace morpheus {
/****** Component public implementations *******************/
/****** CuDFTableUtil****************************************/

/**
 * @addtogroup utilities
 * @{
 * @file
 */

/**
 * @brief Structure that encapsulates cuDF table utilities.
 */
struct MORPHEUS_EXPORT CuDFTableUtil
{
    /**
     * @brief Load a table from a file.
     *
     * @param filename The name of the file to load.
     * @return cudf::io::table_with_metadata The table loaded from the file.
     */
    static cudf::io::table_with_metadata load_table(const std::string& filename);

    /**
     * @brief Get the column names from a cudf table_with_metadata.
     *
     * @param table The table to get the column names from.
     * @return std::vector<std::string> The column names.
     */
    static std::vector<std::string> get_column_names(const cudf::io::table_with_metadata& table);

    /**
     * @brief Filters rows from a table that contain null values in a given columns.
     * null values in columns other than those specified in `filter_columns` are not considered.
     * Any missing columns are ignored.
     *
     * @param table The table to filter
     * @param filter_columns The name of the columns to filter on
     */
    static void filter_null_data(cudf::io::table_with_metadata& table, const std::vector<std::string>& filter_columns);
};
/** @} */  // end of group
}  // namespace morpheus
