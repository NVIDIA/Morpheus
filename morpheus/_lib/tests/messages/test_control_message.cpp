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

#include "nlohmann/json.hpp"
#include "test_messages.hpp"

#include "morpheus/messages/control.hpp"
#include "morpheus/messages/meta.hpp"

#include <memory>

using namespace morpheus;
using namespace morpheus::test;

TEST_F(TestControlMessage, InitializationTest)
{
    auto msg_one = MessageControl();

    auto config          = nlohmann::json();
    config["some_value"] = "42";

    auto msg_two = MessageControl(config);

    ASSERT_EQ(msg_two.config().contains("some_value"), true);
    ASSERT_EQ(msg_two.config()["some_value"], "42");
}

TEST_F(TestControlMessage, SetMessageTest)
{
    auto msg = MessageControl();

    ASSERT_EQ(msg.config().contains("some_value"), false);

    auto config          = nlohmann::json();
    config["some_value"] = "42";

    msg.config(config);

    ASSERT_EQ(msg.config().contains("some_value"), true);
    ASSERT_EQ(msg.config()["some_value"], "42");
}

TEST_F(TestControlMessage, PayloadTest)
{
    auto msg = MessageControl();

    ASSERT_EQ(msg.payload(), nullptr);

    auto null_payload = std::shared_ptr<MessageMeta>(nullptr);

    msg.payload(null_payload);

    ASSERT_EQ(msg.payload(), null_payload);

    auto data_payload = create_mock_msg_meta({"col1", "col2", "col3"}, {"int32", "float32", "string"}, 5);

    msg.payload(data_payload);

    ASSERT_EQ(msg.payload(), data_payload);
}