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

#include "morpheus_llm/llm/llm_context.hpp"  // for LLMContext
#include "morpheus_llm/llm/llm_lambda_node.hpp"
#include "morpheus_llm/llm/llm_node.hpp"  // for LLMNode
#include "morpheus_llm/llm/llm_node_runner.hpp"
#include "morpheus_llm/llm/llm_task.hpp"

#include "morpheus/_lib/tests/test_utils/common.hpp"  // IWYU pragma: associated
#include "morpheus/types.hpp"
#include "morpheus/utilities/json_types.hpp"  // for PythonByteContainer

#include <gtest/gtest.h>
#include <mrc/coroutines/sync_wait.hpp>

#include <coroutine>
#include <memory>
#include <stdexcept>
#include <string>
// IWYU pragma: no_include "morpheus_llm/llm/fwd.hpp"

using namespace morpheus;
using namespace morpheus::test;

TEST_CLASS(LLMNode);

auto make_dummy_node()
{
    return llm::make_lambda_node([]() -> Task<int> {
        co_return 0;
    });
}

auto make_single_input_node()
{
    return llm::make_lambda_node([](int i) -> Task<int> {
        co_return i + 1;
    });
}

TEST_F(TestLLMNode, NoNodes)
{
    llm::LLMNode node;

    auto context = std::make_shared<llm::LLMContext>(llm::LLMTask{}, nullptr);

    auto out_context = mrc::coroutines::sync_wait(node.execute(context));

    ASSERT_EQ(out_context, context);
    ASSERT_EQ(out_context->view_outputs().size(), 0);
}

TEST_F(TestLLMNode, AddNode)
{
    llm::LLMNode node;

    auto runner = node.add_node("Node1", {}, make_dummy_node());

    ASSERT_EQ(runner->name(), "Node1");
    ASSERT_EQ(runner->inputs().size(), 0);
    ASSERT_EQ(node.node_count(), 1);
}

TEST_F(TestLLMNode, AddNodeOutput)
{
    llm::LLMNode node;

    node.add_node("NotOutput", {}, make_dummy_node());

    ASSERT_EQ(node.get_output_node_names(), std::vector<std::string>{});

    auto runner = node.add_node("Output", {}, make_dummy_node(), true);

    ASSERT_EQ(node.get_output_node_names(), std::vector<std::string>{"Output"});
}

TEST_F(TestLLMNode, DuplicateNode)
{
    llm::LLMNode node;

    node.add_node("Node1", {}, make_dummy_node());

    EXPECT_THROW(node.add_node("Node1", {}, make_dummy_node()), std::invalid_argument);
}

TEST_F(TestLLMNode, InputNodeDoesNotExist)
{
    llm::LLMNode node;

    // No upstream nodes
    EXPECT_THROW(node.add_node("DummyNode", {{"/NodeDoesNotExist"}}, make_single_input_node()), std::invalid_argument);
    ASSERT_EQ(node.node_count(), 0);
    ASSERT_EQ(node.get_input_names().size(), 0);
    ASSERT_EQ(node.get_output_node_names().size(), 0);

    node.add_node("Node1", {}, make_dummy_node());

    // Wrong name
    EXPECT_THROW(node.add_node("Node2", {{"/NodeDoesNotExist"}}, make_single_input_node()), std::invalid_argument);
    ASSERT_EQ(node.node_count(), 1);
    ASSERT_EQ(node.get_input_names().size(), 0);
    ASSERT_EQ(node.get_output_node_names().size(), 0);
}

TEST_F(TestLLMNode, InputNameDoesNotExist)
{
    llm::LLMNode node;

    // Mismatched number of inputs
    EXPECT_THROW(node.add_node("Node1", {}, make_single_input_node()), std::invalid_argument);

    EXPECT_THROW(node.add_node("Node1", {{"Test"}}, make_dummy_node()), std::invalid_argument);
}

TEST_F(TestLLMNode, WrongNumberOfNodes)
{
    llm::LLMNode node;

    // Mismatched number of inputs
    EXPECT_THROW(node.add_node("Node1", {}, make_single_input_node()), std::invalid_argument);

    EXPECT_THROW(node.add_node("Node1", {{"Test"}}, make_dummy_node()), std::invalid_argument);
}

TEST_F(TestLLMNode, AddChildNode)
{
    llm::LLMNode node;

    node.add_node("Root1", {}, make_dummy_node());

    auto child_node = std::make_shared<llm::LLMNode>();

    child_node->add_node("Child1", {{"ChildInput"}}, make_single_input_node());

    child_node->add_node("Child2", {{"/Child1"}}, make_single_input_node(), true);

    ASSERT_EQ(child_node->get_input_names(), std::vector<std::string>{"ChildInput"});
    ASSERT_EQ(child_node->get_output_node_names(), std::vector<std::string>{"Child2"});

    node.add_node("Root2", {{"/Root1", "ChildInput"}}, child_node);

    node.add_node("Root3", {{"/Root2"}}, make_single_input_node(), true);

    ASSERT_EQ(node.get_output_node_names(), std::vector<std::string>{"Root3"});

    auto context     = std::make_shared<llm::LLMContext>(llm::LLMTask{}, nullptr);
    auto out_context = mrc::coroutines::sync_wait(node.execute(context));

    ASSERT_EQ(out_context->view_outputs()["Root3"], 3);
}
