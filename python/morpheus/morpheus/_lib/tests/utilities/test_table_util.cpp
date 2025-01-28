/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../test_utils/common.hpp"  // IWYU pragma: associated

#include "morpheus/io/deserializers.hpp"
#include "morpheus/utilities/table_util.hpp"  // for filter_null_data

#include <cudf/io/types.hpp>     // for table_with_metadata
#include <cudf/table/table.hpp>  // for table
#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <utility>  // for pair
#include <vector>
// IWYU pragma: no_include <initializer_list>

using namespace morpheus;

TEST_CLASS(TableUtil);

TEST_F(TestTableUtil, GetColumnNames)
{
    auto morpheus_root = test::get_morpheus_root();
    auto input_files   = {morpheus_root / "tests/tests_data/file_with_nulls.csv",
                          morpheus_root / "tests/tests_data/file_with_nulls.jsonlines"};

    for (const auto& input_file : input_files)
    {
        auto table_w_meta = load_table_from_file(input_file);
        auto column_names = CuDFTableUtil::get_column_names(table_w_meta);

        EXPECT_EQ(column_names.size(), 2);
        EXPECT_EQ(column_names[0], "data");
        EXPECT_EQ(column_names[1], "other");
    }
}

TEST_F(TestTableUtil, FilterNullData)
{
    auto morpheus_root = test::get_morpheus_root();
    auto input_files   = {morpheus_root / "tests/tests_data/file_with_nans.csv",
                          morpheus_root / "tests/tests_data/file_with_nans.jsonlines",
                          morpheus_root / "tests/tests_data/file_with_nulls.csv",
                          morpheus_root / "tests/tests_data/file_with_nulls.jsonlines"};
    std::vector<std::pair<std::vector<std::string>, std::size_t>> expected_row_counts{
        {{"data"}, 8}, {{"data"}, 8}, {{"other"}, 7}, {{"other"}, 7}, {{"data", "other"}, 5}};

    for (const auto& input_file : input_files)
    {
        for (const auto& [filter_columns, expected_row_count] : expected_row_counts)
        {
            auto table_w_meta = load_table_from_file(input_file);

            EXPECT_EQ(table_w_meta.tbl->num_columns(), 2);
            EXPECT_EQ(table_w_meta.tbl->num_rows(), 10);

            CuDFTableUtil::filter_null_data(table_w_meta, filter_columns);

            EXPECT_EQ(table_w_meta.tbl->num_columns(), 2);
            EXPECT_EQ(table_w_meta.tbl->num_rows(), expected_row_count);
        }
    }
}
