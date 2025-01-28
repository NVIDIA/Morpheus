/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "./test_utils/common.hpp"  // IWYU pragma: associated

#include "morpheus/io/deserializers.hpp"

#include <cudf/io/csv.hpp>
#include <cudf/io/types.hpp>
#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

using namespace morpheus;

namespace {
const std::string USER_COLS{",time,eventID,eventSource,eventName,z_loss\n"};
const std::string DATA_ROW{"0,1,2,test,test,5\n"};
}  // namespace

TEST_CLASS(Deserializers);

TEST_F(TestDeserializers, GetIndexColCountNoIdxFromFile)
{
    auto test_data_dir = test::get_morpheus_root() / "tests/tests_data";

    // First test a files without an index
    std::vector<std::filesystem::path> input_files{test_data_dir / "filter_probs.csv",
                                                   test_data_dir / "filter_probs.jsonlines"};
    for (const auto& input_file : input_files)
    {
        auto table = load_table_from_file(input_file);
        EXPECT_EQ(get_index_col_count(table), 0);
    }
}

TEST_F(TestDeserializers, GetIndexColCountWithIdxFromFile)
{
    // now test a file with an index
    auto input_file = test::get_morpheus_root() / "tests/tests_data/filter_probs_w_id_col.csv";
    auto table      = load_table_from_file(input_file);
    EXPECT_EQ(get_index_col_count(table), 1);
}

TEST_F(TestDeserializers, GetIndexColCountNoIdxSimilarName)
{
    std::vector<std::string> not_id_cols{"identity", "Unnamed: 01", "test id", "email"};

    for (const auto& col : not_id_cols)
    {
        // Build a list of column names with `col` as our first element followed by the columns in `UserCols`
        std::string csv_data{col + USER_COLS + DATA_ROW};

        auto options = cudf::io::csv_reader_options::builder(cudf::io::source_info{csv_data.c_str(), csv_data.size()});
        auto table   = cudf::io::read_csv(options.build());

        EXPECT_EQ(get_index_col_count(table), 0);
    }
}

TEST_F(TestDeserializers, GetIndexColCountIdx)
{
    // These should all match regardless of whitespace and case
    std::vector<std::string> id_cols{
        "Unnamed: 0", "UNNamed: 0", "Unnamed: 0\t", "\tUnnamed: 0\t", " Unnamed: 0", "ID", "id", "id ", " id", " iD "};

    for (const auto& col : id_cols)
    {
        // Build a list of column names with `col` as our first element followed by the columns in `UserCols`
        std::string csv_data{col + USER_COLS + DATA_ROW};

        auto options = cudf::io::csv_reader_options::builder(cudf::io::source_info{csv_data.c_str(), csv_data.size()});
        auto table   = cudf::io::read_csv(options.build());

        EXPECT_EQ(get_index_col_count(table), 1);
    }
}

TEST_F(TestDeserializers, GetIndexColCountValidNameInvalidType)
{
    // Construct a csv with two rows, first row will contain the expected int id value second row will be a string
    std::string data_rows{DATA_ROW + "s,6,7,frog,toad,8\n"};
    std::string csv_data{"id" + USER_COLS + data_rows};

    auto options = cudf::io::csv_reader_options::builder(cudf::io::source_info{csv_data.c_str(), csv_data.size()});
    auto table   = cudf::io::read_csv(options.build());

    EXPECT_EQ(get_index_col_count(table), 0);
}
