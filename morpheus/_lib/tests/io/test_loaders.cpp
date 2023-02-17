/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "test_io.hpp"

#include "morpheus/io/loaders/all.hpp"
#include "morpheus/messages/control.hpp"

#include <fstream>
#include <iostream>

namespace py = pybind11;
using namespace morpheus;
using namespace morpheus::test;

TEST_F(TestLoader, LoaderInitializationTest)
{
    auto file    = FileDataLoader();
    auto grpc    = GRPCDataLoader();
    auto payload = PayloadDataLoader();
    auto rest    = RESTDataLoader();
}

TEST_F(TestLoader, LoaderFileTest)
{
    auto string_df = create_mock_csv_file({"col1", "col2", "col3"}, {"int32", "float32", "string"}, 5);

    char temp_file[] = "/tmp/morpheus_test_XXXXXXXX";
    int fd           = mkstemp(temp_file);
    if (fd == -1)
    {
        GTEST_SKIP() << "Failed to create temporary file, skipping test";
    }

    nlohmann::json config;
    config["loader_id"] = "file";
    config["strategy"]  = "aggregate";
    config["files"]     = nlohmann::json::array();

    config["files"].push_back({{"path", std::string(temp_file)}, {"type", "csv"}});

    std::fstream data_file(temp_file, std::ios::out | std::ios::binary | std::ios::trunc);
    data_file << string_df;
    data_file.close();

    auto msg    = MessageControl(config);
    auto loader = FileDataLoader();

    EXPECT_NO_THROW(loader.load(msg));

    unlink(temp_file);
}

TEST_F(TestLoader, LoaderGRPCTest)
{
    auto msg    = MessageControl();
    auto loader = GRPCDataLoader();

    EXPECT_THROW(loader.load(msg), std::runtime_error);
}

TEST_F(TestLoader, LoaderPayloadTest)
{
    auto msg    = MessageControl();
    auto loader = PayloadDataLoader();

    EXPECT_NO_THROW(loader.load(msg));
}

TEST_F(TestLoader, LoaderRESTTest)
{
    auto msg    = MessageControl();
    auto loader = RESTDataLoader();

    EXPECT_THROW(loader.load(msg), std::runtime_error);
}
