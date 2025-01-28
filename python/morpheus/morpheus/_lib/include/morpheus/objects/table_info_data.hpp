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

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>  // for size_type

#include <numeric>
#include <string>
#include <vector>

namespace morpheus {

/**
 * @brief Simple structure which provides a general method for holding a cudf:table_view together with index and column
 * names.
 *
 * @note This struct should always be header only since its the only transfer mechanism between libmorpheus.so and
 * cudf_helpers.
 *
 */
struct TableInfoData
{
    TableInfoData() = default;
    TableInfoData(cudf::table_view view, std::vector<std::string> indices, std::vector<std::string> columns) :
      table_view(std::move(view)),
      index_names(std::move(indices)),
      column_names(std::move(columns))
    {
        std::vector<cudf::size_type> col_indices(column_names.size());
        std::iota(col_indices.begin(), col_indices.end(), 0);
        column_indices = std::move(col_indices);
    }
    TableInfoData(cudf::table_view view,
                  std::vector<std::string> indices,
                  std::vector<std::string> columns,
                  std::vector<cudf::size_type> col_indicies) :
      table_view(std::move(view)),
      index_names(std::move(indices)),
      column_names(std::move(columns)),
      column_indices(std::move(col_indicies))
    {}

    cudf::table_view table_view;
    std::vector<std::string> index_names;
    std::vector<std::string> column_names;
    std::vector<cudf::size_type> column_indices;
};

}  // namespace morpheus
