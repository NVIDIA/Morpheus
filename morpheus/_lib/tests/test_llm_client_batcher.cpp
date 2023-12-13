/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "./test_utils/common.hpp"

#include "morpheus/objects/llm_client_batcher.hpp"

#include <gtest/gtest.h>

#include <memory>

TEST_CLASS(LLMClientBatcher);

TEST_F(TestLLMClientBatcher, Simple)
{
    auto batcher = LLMClientBatcher();

    auto response = batcher.generate({"a", "b"});

    auto result = response.get();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0], "cool story");
    EXPECT_EQ(result[1], "cool story");

    auto response2 = batcher.generate({"a"});
    auto response3 = batcher.generate({"a", "b", "c"});

    auto result2 = response2.get();
    EXPECT_EQ(result2.size(), 1);
    EXPECT_EQ(result2[0], "cool story");

    auto result3 = response3.get();
    EXPECT_EQ(result3.size(), 3);
    EXPECT_EQ(result3[0], "cool story");
    EXPECT_EQ(result3[1], "cool story");
    EXPECT_EQ(result3[2], "cool story");
}
