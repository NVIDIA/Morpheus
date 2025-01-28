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

#include "morpheus_llm/llm/llm_task.hpp"

#include "morpheus/_lib/tests/test_utils/common.hpp"  // IWYU pragma: associated

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

using namespace morpheus;
using namespace morpheus::test;

TEST_CLASS(LLMTask);

TEST_F(TestLLMTask, Initialization)
{
    llm::LLMTask task_1;
    ASSERT_EQ(task_1.size(), 0);

    nlohmann::json task_dict;
    task_dict = {
        {"task_type", "dictionary"},
        {"model_name", "test1"},
    };

    llm::LLMTask task_2{"template", task_dict};
    ASSERT_EQ(task_2.get("task_type"), "dictionary");
    ASSERT_EQ(task_2.get("model_name"), "test1");
    ASSERT_EQ(task_2.size(), 2);
}

TEST_F(TestLLMTask, SetTaskDict)
{
    llm::LLMTask task_1;

    nlohmann::json task_dict;
    task_dict = {
        {"task_type", "dictionary"},
        {"model_name", "test1"},
    };

    llm::LLMTask task_2{"template", task_dict};

    task_2.set("model_name", "test2");
    task_2.set("addkey_1", "val1");
    task_2.set("addkey_2", "val2");
    task_2.set("addkey_3", "val3");
    ASSERT_EQ(task_2.get("task_type"), "dictionary");
    ASSERT_EQ(task_2.get("model_name"), "test2");
    ASSERT_EQ(task_2.get("addkey_1"), "val1");
    ASSERT_EQ(task_2.get("addkey_2"), "val2");
    ASSERT_EQ(task_2.get("addkey_3"), "val3");
    ASSERT_EQ(task_2.size(), 5);
}
