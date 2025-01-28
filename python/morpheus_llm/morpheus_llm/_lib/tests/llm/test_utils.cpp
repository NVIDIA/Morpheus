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

#include "morpheus_llm/llm/fwd.hpp"  // for morpheus
#include "morpheus_llm/llm/input_map.hpp"
#include "morpheus_llm/llm/utils.hpp"

#include "morpheus/_lib/tests/test_utils/common.hpp"  // IWYU pragma: associated
#include "morpheus/types.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace morpheus;
using namespace morpheus::test;
using namespace mrc;

TEST_CLASS(LLMUtils);

TEST_F(TestLLMUtils, IsValidNameEmpty)
{
    EXPECT_FALSE(llm::is_valid_node_name(""));
}

TEST_F(TestLLMUtils, IsValidNameSpecialCharacters)
{
    std::vector<std::string> check_list{"my_name*", "my_name/stuff", "my_name[0]", "Piñata", "Kulübü", "漢字"};
    for (const auto& name : check_list)
    {
        EXPECT_FALSE(llm::is_valid_node_name(name));
    }
}

TEST_F(TestLLMUtils, IsValidNameSpace)
{
    std::vector<std::string> check_list{"my_name other_name",
                                        "my_name\tother_name",
                                        "my_name\nother_name",
                                        "my_name\rother_name",
                                        " prefix",
                                        "trailing "};
    for (const auto& name : check_list)
    {
        EXPECT_FALSE(llm::is_valid_node_name(name));
    }
}

TEST_F(TestLLMUtils, ProcessInputNamesSingleMapping)
{
    auto user_inputs = llm::user_input_mappings_t{{"/ext1", "input1"}};
    auto input_names = std::vector<std::string>{"input1"};

    auto returned = llm::process_input_names(user_inputs, input_names);

    EXPECT_EQ(returned.size(), 1);
    EXPECT_EQ(returned[0].external_name, "/ext1");
    EXPECT_EQ(returned[0].internal_name, "input1");
}

TEST_F(TestLLMUtils, ProcessInputNamesSingleString)
{
    std::vector<std::vector<std::string>> mappings{{"/ext1", "input1"}, {"input1", "input1"}};
    for (const auto& m : mappings)
    {
        auto user_inputs = llm::user_input_mappings_t{{m[0]}};
        auto input_names = std::vector<std::string>{m[1]};

        auto returned = llm::process_input_names(user_inputs, input_names);
        EXPECT_EQ(returned.size(), 1);
        EXPECT_EQ(returned[0].external_name, m[0]);
        EXPECT_EQ(returned[0].internal_name, m[1]);
    }
}

TEST_F(TestLLMUtils, ProcessInputNamesMultipleMappings)
{
    auto user_inputs = llm::user_input_mappings_t{{"/ext1", "input1"}, {"/ext2", "input2"}};
    auto input_names = std::vector<std::string>{"input1", "input2"};

    auto returned = llm::process_input_names(user_inputs, input_names);

    EXPECT_EQ(returned.size(), 2);
    EXPECT_EQ(returned[0].external_name, "/ext1");
    EXPECT_EQ(returned[0].internal_name, "input1");
    EXPECT_EQ(returned[1].external_name, "/ext2");
    EXPECT_EQ(returned[1].internal_name, "input2");
}

TEST_F(TestLLMUtils, ProcessInputNamesMismatchSingleMapping)
{
    auto user_inputs = llm::user_input_mappings_t{{"/ext1", "input2"}};
    auto input_names = std::vector<std::string>{"input1"};

    EXPECT_THROW(llm::process_input_names(user_inputs, input_names), std::invalid_argument);
}

TEST_F(TestLLMUtils, ProcessInputNamesMismatchMultipleMappings)
{
    auto user_inputs = llm::user_input_mappings_t{{"/ext1", "input1"}, {"/ext2", "input3"}};
    auto input_names = std::vector<std::string>{"input1", "input2"};

    EXPECT_THROW(llm::process_input_names(user_inputs, input_names), std::invalid_argument);
}

TEST_F(TestLLMUtils, ProcessInputNamesCountMismatch)
{
    auto user_inputs = llm::user_input_mappings_t{{"/ext1", "input1"}, {"/ext2", "input1"}};

    EXPECT_THROW(llm::process_input_names(user_inputs, std::vector<std::string>{"input1", "input2"}),
                 std::invalid_argument);
    EXPECT_THROW(llm::process_input_names(user_inputs, std::vector<std::string>{"input1"}), std::invalid_argument);
}

TEST_F(TestLLMUtils, ProcessInputNamesPlaceholderSingleInput)
{
    auto user_inputs = llm::user_input_mappings_t{{"/ext1/*", "*"}};
    auto input_names = std::vector<std::string>{"input1"};

    auto returned = llm::process_input_names(user_inputs, input_names);

    EXPECT_EQ(returned.size(), 1);
    EXPECT_EQ(returned[0].external_name, "/ext1/input1");
    EXPECT_EQ(returned[0].internal_name, "input1");
}

TEST_F(TestLLMUtils, ProcessInputNamesPlaceholderMultipleInputs)
{
    auto user_inputs = llm::user_input_mappings_t{{"/ext1/*", "*"}};
    auto input_names = std::vector<std::string>{"input1, input2"};

    auto returned = llm::process_input_names(user_inputs, input_names);

    EXPECT_EQ(returned.size(), 1);
    EXPECT_EQ(returned[0].external_name, "/ext1/input1, input2");
    EXPECT_EQ(returned[0].internal_name, "input1, input2");
}

TEST_F(TestLLMUtils, ProcessInputNamesPlaceholderMismatch)
{
    auto input_names = std::vector<std::string>{"input1"};

    EXPECT_THROW(llm::process_input_names(llm::user_input_mappings_t{{"/ext1", "*"}}, input_names),
                 std::invalid_argument);
    EXPECT_THROW(llm::process_input_names(llm::user_input_mappings_t{{"/ext1/*", "input1"}}, input_names),
                 std::invalid_argument);
}

TEST_F(TestLLMUtils, ProcessInputNamesIndexMatching)
{
    auto user_inputs = llm::user_input_mappings_t{{"/ext1"}, {"/ext2"}};
    auto input_names = std::vector<std::string>{"input1", "input2"};

    auto returned = llm::process_input_names(user_inputs, input_names);

    EXPECT_EQ(returned.size(), 2);
    EXPECT_EQ(returned[0].external_name, "/ext1");
    EXPECT_EQ(returned[0].internal_name, "input1");
    EXPECT_EQ(returned[1].external_name, "/ext2");
    EXPECT_EQ(returned[1].internal_name, "input2");
}

TEST_F(TestLLMUtils, ProcessInputNamesNameMatching)
{
    auto user_inputs = llm::user_input_mappings_t{{"input1"}, {"input2"}};
    auto input_names = std::vector<std::string>{"input1", "input2"};

    auto returned = llm::process_input_names(user_inputs, input_names);

    EXPECT_EQ(returned.size(), 2);
    EXPECT_EQ(returned[0].external_name, "input1");
    EXPECT_EQ(returned[0].internal_name, "input1");
    EXPECT_EQ(returned[1].external_name, "input2");
    EXPECT_EQ(returned[1].internal_name, "input2");
}

TEST_F(TestLLMUtils, ProcessInputNamesMixNameAndIndexMatching)
{
    auto user_inputs = llm::user_input_mappings_t{{"input1"}, {"/ext1"}};
    auto input_names = std::vector<std::string>{"input1", "input2"};

    EXPECT_THROW(llm::process_input_names(user_inputs, input_names), std::invalid_argument);
}

TEST_F(TestLLMUtils, ProcessInputNamesSiblingWithInputNameMatching)
{
    auto user_inputs = llm::user_input_mappings_t{{"/ext1/input1"}};
    auto input_names = std::vector<std::string>{"input1"};

    auto returned = llm::process_input_names(user_inputs, input_names);

    EXPECT_EQ(returned.size(), 1);
    EXPECT_EQ(returned[0].external_name, "/ext1/input1");
    EXPECT_EQ(returned[0].internal_name, "input1");
}

TEST_F(TestLLMUtils, ProcessInputNamesMatchingIndexExceeded)
{
    auto input_names = std::vector<std::string>{"input1"};

    EXPECT_THROW(llm::process_input_names(llm::user_input_mappings_t{{"/ext1"}, {"/ext2"}}, input_names),
                 std::invalid_argument);
    EXPECT_THROW(llm::process_input_names(llm::user_input_mappings_t{{"input1"}, {"input2"}}, input_names),
                 std::invalid_argument);
}
