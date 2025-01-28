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
#include "test_messages.hpp"         // for TestMessages

#include "morpheus/messages/control.hpp"               // for ControlMessage
#include "morpheus/messages/memory/tensor_memory.hpp"  // for TensorMemory
#include "morpheus/messages/meta.hpp"                  // for MessageMeta
#include "morpheus/types.hpp"
#include "morpheus/utilities/json_types.hpp"           // for PythonByteContainer

#include <gtest/gtest.h>       // for Message, TestPartResult, AssertionResult, TestInfo
#include <nlohmann/json.hpp>   // for basic_json, json_ref, json
#include <pybind11/pytypes.h>  // for literals, pybind11
#include <pybind11/stl.h>      // IWYU pragma: keep

#include <algorithm>  // for find
#include <chrono>     // for system_clock
#include <map>        // for map
#include <memory>     // for allocator, make_shared, shared_ptr
#include <optional>   // for optional
#include <stdexcept>  // for runtime_error
#include <string>     // for operator<=>, string, char_traits, basic_string
#include <vector>     // for vector

using namespace morpheus;
using namespace morpheus::test;

using clock_type_t = std::chrono::system_clock;

using TestControlMessage = morpheus::test::TestMessages;  // NOLINT(readability-identifier-naming)

namespace py = pybind11;
using namespace pybind11::literals;
using namespace std::string_literals;

TEST_F(TestControlMessage, InitializationTest)
{
    auto msg_one = ControlMessage();

    auto config = nlohmann::json();
    nlohmann::json task_properties;
    task_properties = {
        {"loader_id", "payload"},
        {"strategy", "aggregate"},
    };
    config["tasks"] = {{{"type", "load"}, {"properties", task_properties}}};

    auto msg_two = ControlMessage(config);

    ASSERT_EQ(msg_two.has_task("load"), true);
}

TEST_F(TestControlMessage, SetAndGetMetadata)
{
    auto msg = ControlMessage();

    nlohmann::json value = {{"property", "value"}};
    std::string key      = "testKey";

    // Set metadata
    msg.set_metadata(key, value);

    // Verify metadata can be retrieved and matches what was set
    EXPECT_TRUE(msg.has_metadata(key));
    auto retrievedValue = msg.get_metadata(key, true);
    EXPECT_EQ(value, retrievedValue);

    // Verify listing metadata includes the key
    auto keys = msg.list_metadata();
    auto it   = std::find(keys.begin(), keys.end(), key);
    EXPECT_NE(it, keys.end());
}

// Test for overwriting metadata
TEST_F(TestControlMessage, OverwriteMetadata)
{
    auto msg = ControlMessage();

    nlohmann::json value1 = {{"initial", "data"}};
    nlohmann::json value2 = {{"updated", "data"}};
    std::string key       = "overwriteKey";

    // Set initial metadata
    msg.set_metadata(key, value1);

    // Overwrite metadata
    msg.set_metadata(key, value2);

    // Verify metadata was overwritten
    auto retrievedValue = msg.get_metadata(key, false);
    EXPECT_EQ(value2, retrievedValue);
}

// Test retrieving metadata when it does not exist
TEST_F(TestControlMessage, GetNonexistentMetadata)
{
    auto msg = ControlMessage();

    std::string key = "nonexistentKey";

    // Attempt to retrieve metadata that does not exist
    EXPECT_FALSE(msg.has_metadata(key));
    EXPECT_THROW(auto const x = msg.get_metadata(key, true), std::runtime_error);
    EXPECT_NO_THROW(auto const x = msg.get_metadata(key, false));  // Should not throw, but return empty json
}

// Test retrieving all metadata
TEST_F(TestControlMessage, GetAllMetadata)
{
    auto msg = ControlMessage();

    // Setup - add some metadata
    msg.set_metadata("key1", {{"data", "value1"}});
    msg.set_metadata("key2", {{"data", "value2"}});

    // Retrieve all metadata
    auto metadata = msg.get_metadata();
    EXPECT_EQ(2, metadata.size());  // Assuming get_metadata() returns a json object with all metadata
    EXPECT_TRUE(metadata.contains("key1"));
    EXPECT_TRUE(metadata.contains("key2"));
}

TEST_F(TestControlMessage, SetMessageTest)
{
    auto msg = ControlMessage();

    ASSERT_EQ(msg.config().contains("nope"), false);

    auto config = nlohmann::json();
    nlohmann::json task_properties;
    task_properties = {
        {"loader_id", "payload"},
        {"strategy", "aggregate"},
    };
    config["tasks"] = {{{"type", "load"}, {"properties", task_properties}}};

    msg.config(config);

    ASSERT_EQ(msg.has_task("load"), true);
}

TEST_F(TestControlMessage, TaskTest)
{
    auto msg_infer = ControlMessage();
    auto msg_train = ControlMessage();

    ASSERT_EQ(msg_infer.config().contains("some_value"), false);
    auto config = nlohmann::json();
    nlohmann::json task_properties;
    task_properties = {
        {"loader_id", "payload"},
        {"strategy", "aggregate"},
    };

    config["type"]  = "inference";
    config["tasks"] = {{{"type", "load"}, {"properties", task_properties}}};

    msg_infer.config(config);

    ASSERT_EQ(msg_infer.has_task("load"), true);
    ASSERT_EQ(msg_infer.has_task("inference"), false);
    ASSERT_EQ(msg_infer.has_task("training"), false);
    ASSERT_EQ(msg_infer.has_task("custom"), false);

    msg_infer.add_task("inference", {});
    ASSERT_EQ(msg_infer.has_task("inference"), true);

    msg_infer.remove_task("inference");
    ASSERT_EQ(msg_infer.has_task("inference"), false);

    ASSERT_THROW(msg_infer.add_task("training", {}), std::runtime_error);

    config["type"] = "training";
    msg_train.config(config);
    msg_train.add_task("training", {});
    ASSERT_EQ(msg_train.has_task("training"), true);
    msg_train.remove_task("training");
    ASSERT_EQ(msg_train.has_task("training"), false);

    ASSERT_THROW(msg_train.add_task("inference", {}), std::runtime_error);

    msg_train.add_task("custom", {});
    ASSERT_EQ(msg_train.has_task("custom"), true);
    msg_train.remove_task("custom");
    ASSERT_EQ(msg_train.has_task("custom"), false);
}

TEST_F(TestControlMessage, PayloadTest)
{
    auto msg = ControlMessage();

    ASSERT_EQ(msg.payload(), nullptr);

    auto null_payload = std::shared_ptr<MessageMeta>(nullptr);

    msg.payload(null_payload);

    ASSERT_EQ(msg.payload(), null_payload);

    auto data_payload = create_mock_msg_meta({"col1", "col2", "col3"}, {"int32", "float32", "string"}, 5);

    msg.payload(data_payload);

    ASSERT_EQ(msg.payload(), data_payload);
}

TEST_F(TestControlMessage, SetAndGetTimestamp)
{
    auto msg = ControlMessage();

    // Test setting a timestamp
    auto start = clock_type_t::now();
    msg.set_timestamp("group1::key1", start);

    auto result = msg.get_timestamp("group1::key1", false);
    ASSERT_TRUE(result.has_value());

    // Direct comparison since we're using time points now
    EXPECT_EQ(start, result.value());
}

TEST_F(TestControlMessage, GetTimestampWithRegex)
{
    auto start = clock_type_t::now();
    auto msg   = ControlMessage();

    // Set two timestamps slightly apart
    msg.set_timestamp("group1::key1", start);
    auto later = clock_type_t::now();
    msg.set_timestamp("group1::key2", later);

    auto result = msg.filter_timestamp("group1::key.*");
    ASSERT_EQ(2, result.size());

    // Check using the actual time points
    EXPECT_EQ(start, result["group1::key1"]);
    EXPECT_EQ(later, result["group1::key2"]);

    auto resultSingle = msg.filter_timestamp("group1::key1");
    ASSERT_EQ(1, resultSingle.size());
    EXPECT_EQ(start, resultSingle["group1::key1"]);
}

TEST_F(TestControlMessage, GetTimestampNonExistentKey)
{
    auto msg = ControlMessage();

    auto result = msg.get_timestamp("group1::nonexistent", false);
    EXPECT_FALSE(result.has_value());

    EXPECT_THROW(
        {
            try
            {
                msg.get_timestamp("group1::nonexistent", true);
            } catch (const std::runtime_error& e)
            {
                EXPECT_STREQ("Timestamp for the specified key does not exist.", e.what());
                throw;
            }
        },
        std::runtime_error);
}

TEST_F(TestControlMessage, UpdateTimestamp)
{
    auto msg = ControlMessage();

    auto start = clock_type_t::now();
    msg.set_timestamp("group1::key1", start);
    auto later = clock_type_t::now();
    msg.set_timestamp("group1::key1", later);

    auto result = msg.get_timestamp("group1::key1", false);
    ASSERT_TRUE(result.has_value());

    // Check using the actual time points for update
    EXPECT_EQ(later, result.value());
}

// Test setting and getting Ten:sorMemory
TEST_F(TestControlMessage, SetAndGetTensorMemory)
{
    auto msg = ControlMessage();

    TensorIndex count = 5;
    auto tensorMemory = std::make_shared<TensorMemory>(count);
    // Optionally, modify tensorMemory here if it has any mutable state to test

    // Set the tensor memory
    msg.tensors(tensorMemory);

    // Retrieve the tensor memory
    auto retrievedTensorMemory = msg.tensors();

    // Verify that the retrieved tensor memory matches what was set
    EXPECT_EQ(tensorMemory, retrievedTensorMemory);
    EXPECT_EQ(count, msg.tensor_count());
}

// Test setting TensorMemory to nullptr
TEST_F(TestControlMessage, SetTensorMemoryToNull)
{
    auto msg = ControlMessage();

    // Set tensor memory to a valid object first
    msg.tensors(std::make_shared<TensorMemory>(0));

    // Now set it to nullptr
    msg.tensors(nullptr);

    // Retrieve the tensor memory
    auto retrievedTensorMemory = msg.tensors();

    // Verify that the retrieved tensor memory is nullptr
    EXPECT_EQ(nullptr, retrievedTensorMemory);
}

// Test retrieving TensorMemory when none has been set
TEST_F(TestControlMessage, GetTensorMemoryWhenNoneSet)
{
    auto msg = ControlMessage();

    // Attempt to retrieve tensor memory without setting it first
    auto retrievedTensorMemory = msg.tensors();

    // Verify that the retrieved tensor memory is nullptr
    EXPECT_EQ(nullptr, retrievedTensorMemory);
}
