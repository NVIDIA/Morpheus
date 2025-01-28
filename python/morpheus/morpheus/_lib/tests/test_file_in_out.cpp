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
#include "morpheus/io/serializers.hpp"
#include "morpheus/messages/meta.hpp"

#include <boost/algorithm/string.hpp>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <pybind11/gil.h>       // for gil_scoped_release, gil_scoped_acquire
#include <pybind11/pybind11.h>  // IWYU pragma: keep

#include <cstddef>
#include <filesystem>
#include <fstream>  // IWYU pragma: keep
#include <memory>   // for shared_ptr
#include <string>
#include <utility>  // for move
#include <vector>
// IWYU pragma: no_include <initializer_list>

using namespace morpheus;

std::string read_file(const std::filesystem::path& file_path)
{
    std::fstream in_stream{file_path, in_stream.in};
    std::stringstream buff;
    in_stream >> buff.rdbuf();
    return buff.str();
}

class TestFileInOut : public morpheus::test::TestWithPythonInterpreter
{};

TEST_F(TestFileInOut, RoundTripCSV)
{
    auto test_data_dir = test::get_morpheus_root() / "tests/tests_data";

    std::vector<std::filesystem::path> input_files{test_data_dir / "filter_probs.csv",
                                                   test_data_dir / "filter_probs_w_id_col.csv"};

    for (const auto& input_file : input_files)
    {
        auto table           = load_table_from_file(input_file);
        auto index_col_count = prepare_df_index(table);

        auto meta = MessageMeta::create_from_cpp(std::move(table), index_col_count);

        pybind11::gil_scoped_release no_gil;
        auto table_info = meta->get_info();
        auto csv_data   = df_to_csv(table_info, true, index_col_count > 0);
        auto src_data   = read_file(input_file);

        boost::trim(csv_data);
        boost::trim(src_data);

        EXPECT_EQ(csv_data, src_data);
    }
}

TEST_F(TestFileInOut, RoundTripJSONLines)
{
    using nlohmann::json;
    auto input_file      = test::get_morpheus_root() / "tests/tests_data/filter_probs.jsonlines";
    auto table           = load_table_from_file(input_file);
    auto index_col_count = prepare_df_index(table);

    EXPECT_EQ(index_col_count, 0);

    auto meta = MessageMeta::create_from_cpp(std::move(table), index_col_count);

    pybind11::gil_scoped_release no_gil;
    auto output_str = df_to_json(meta->get_info());
    boost::trim(output_str);
    std::vector<std::string> output_lines;
    boost::split(output_lines, output_str, boost::is_any_of("\n"));

    auto src_str = read_file(input_file);
    boost::trim(src_str);
    std::vector<std::string> src_lines;
    boost::split(src_lines, src_str, boost::is_any_of("\n"));

    EXPECT_EQ(output_lines.size(), src_lines.size());

    for (std::size_t i = 0; i < output_lines.size(); ++i)
    {
        // Two JSON strings might be equivelant even if the strings are not. ("{\"a\": 5}"" != "{\"a\":5}")
        const auto output_data = json::parse(output_lines[i]);
        const auto src_data    = json::parse(src_lines[i]);

        EXPECT_EQ(output_data, src_data);
    }
}
