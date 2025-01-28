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

#pragma once

#include "morpheus/objects/table_info.hpp"
#include "morpheus/objects/table_info_data.hpp"

#include <cudf/io/types.hpp>
#include <pybind11/pytypes.h>

namespace morpheus {
/****** Component public free function implementations******/

/**
 * @addtogroup utilities
 * @{
 * @file
 */

/**
 * @brief These proxy functions allow us to have a shared set of cudf_helpers interfaces declarations, which proxy
 * the actual generated cython calls. The cython implementation in 'cudf_helpers_api.h' can only appear in the
 * translation unit for the pybind module declaration. These functions should be considered de
 */
struct CudfHelper
{
  public:
    static void load();

    /**
     * @brief Converts a C++ table to a Python DataTable object
     *
     * @param table C++ table with metadata
     * @param index_col_count Number of index columns in the table_with_metadata
     * @return pybind11::object
     */
    static pybind11::object table_from_table_with_metadata(cudf::io::table_with_metadata&& table, int index_col_count);

    /**
     * @brief Converts a C++ TableInfo (which is a view into a python table) into a Python DataTable object
     *
     * @param table_info C++ table view
     * @return pybind11::object
     */
    static pybind11::object table_from_table_info(const morpheus::TableInfoBase& table_info);

    /**
     * @brief Converts a Python DataTable object into a C++ TableInfoData struct containing the column and index
     * information
     *
     * @param table A Python DataTable instance
     * @return TableInfoData
     */
    static TableInfoData table_info_data_from_table(pybind11::object table);

  private:
    CudfHelper();
};

/** @} */  // end of group

}  // namespace morpheus
