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

#include "../test_utils/common.hpp"  // IWYU pragma: associated
#include "test_io.hpp"

#include "morpheus/io/data_loader.hpp"
#include "morpheus/io/loaders/file.hpp"
#include "morpheus/io/loaders/payload.hpp"
#include "morpheus/messages/control.hpp"
#include "morpheus/utilities/json_types.hpp"  // for PythonByteContainer

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <pybind11/embed.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace morpheus;
using namespace morpheus::test;

TEST_F(TestDataLoader, DataLoaderInitializationTest)
{
    auto data_loader = DataLoader();
}

TEST_F(TestDataLoader, DataLoaderRegisterLoaderTest)
{
    auto data_loader = DataLoader();

    nlohmann::json message_config;
    message_config["tasks"] = {{{"type", "load"}, {"properties", {{"loader_id", "payload"}}}}};

    std::vector<std::string> loaders = {"payload"};
    for (auto& loader : loaders)
    {
        auto msg = std::make_shared<ControlMessage>(message_config);

        EXPECT_THROW(data_loader.load(msg), std::runtime_error);
    }

    data_loader.add_loader("payload", std::make_unique<PayloadDataLoader>());

    for (auto& loader : loaders)
    {
        auto msg = std::make_shared<ControlMessage>(message_config);

        EXPECT_NO_THROW(data_loader.load(msg));
    }
}

TEST_F(TestDataLoader, DataLoaderRemoveLoaderTest)
{
    auto data_loader = DataLoader();

    nlohmann::json task_properties;
    task_properties = {{"loader_id", "payload"}};

    auto msg = std::make_shared<ControlMessage>();

    // Load should fail if there are no loaders registered
    msg->add_task("load", task_properties);
    EXPECT_THROW(data_loader.load(msg), std::runtime_error);

    data_loader.add_loader("payload", std::make_unique<PayloadDataLoader>());

    // Load should succeed if there is a loader registered
    msg->add_task("load", task_properties);
    EXPECT_NO_THROW(data_loader.load(msg));

    // Load should fail if the loader is removed
    msg->add_task("load", task_properties);
    data_loader.remove_loader("payload");
    EXPECT_THROW(data_loader.load(msg), std::runtime_error);

    // Shouldn't fail, because there shouldn't be any load tasks on the control message
    EXPECT_NO_THROW(data_loader.load(msg));
}

/**
 * @brief Check that we can send a control message, with a raw data payload and load it correctly.
 */
TEST_F(TestDataLoader, PayloadLoaderTest)
{
    auto data_loader = DataLoader();
    data_loader.add_loader("payload", std::make_unique<PayloadDataLoader>());

    nlohmann::json message_config;
    message_config["tasks"] = {{{"type", "load"},
                                {"properties",
                                 {
                                     {"loader_id", "payload"},
                                 }}}};

    auto msg = std::make_shared<ControlMessage>(message_config);

    auto mm = create_mock_msg_meta({"col1", "col2", "col3"}, {"int32", "float32", "string"}, 5);
    msg->payload(mm);

    auto msg2 = data_loader.load(msg);
    auto mm2  = msg2->payload();
    EXPECT_EQ(mm, mm2);
}

/**
 * @brief Check that we can send a control message, with a raw data payload and load it correctly.
 */
TEST_F(TestDataLoader, FileLoaderTest)
{
    auto data_loader = DataLoader();
    data_loader.add_loader("file", std::make_unique<FileDataLoader>());

    auto string_df = create_mock_csv_file({"col1", "col2", "col3"}, {"int32", "float32", "string"}, 5);

    char temp_file[] = "/tmp/morpheus_test_XXXXXXXX";  // NOLINT
    int fd           = mkstemp(temp_file);
    if (fd == -1)
    {
        GTEST_SKIP() << "Failed to create temporary file, skipping test";
    }

    nlohmann::json message_config;
    message_config["tasks"] = {{{"type", "load"},
                                {"properties",
                                 {
                                     {"loader_id", "file"},
                                     {"strategy", "aggregate"},
                                     {"files",
                                      {
                                          {{"path", std::string(temp_file)}, {"type", "csv"}},
                                      }},
                                 }}}};

    auto msg = std::make_shared<ControlMessage>(message_config);

    std::fstream data_file(temp_file, std::ios::out | std::ios::binary | std::ios::trunc);
    data_file << string_df;
    data_file.close();

    auto mm2 = data_loader.load(msg);
    unlink(temp_file);
}
