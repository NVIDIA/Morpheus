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

#include "morpheus_llm/llm/input_map.hpp"
#include "morpheus_llm/llm/llm_context.hpp"  // for LLMContext
#include "morpheus_llm/llm/llm_lambda_node.hpp"
#include "morpheus_llm/llm/llm_node.hpp"  // for LLMNode
#include "morpheus_llm/llm/llm_node_runner.hpp"
#include "morpheus_llm/llm/llm_task.hpp"

#include "morpheus/_lib/tests/test_utils/common.hpp"  // IWYU pragma: associated
#include "morpheus/types.hpp"
#include "morpheus/utilities/json_types.hpp"  // for PythonByteContainer

#include <gtest/gtest.h>
#include <mrc/coroutines/sync_wait.hpp>  // for sync_wait
#include <mrc/coroutines/when_all.hpp>   // for when_all

#include <coroutine>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>  // for tuple
// IWYU pragma: no_include "morpheus_llm/llm/fwd.hpp"

using namespace morpheus;
using namespace morpheus::test;

TEST_CLASS(LLMNodeRunner);

auto make_nr_dummy_node()
{
    return llm::make_lambda_node([]() -> Task<int> {
        co_return 0;
    });
}

auto make_nr_single_input_node()
{
    return llm::make_lambda_node([](int i) -> Task<int> {
        co_return i + 1;
    });
}

TEST_F(TestLLMNodeRunner, ParentInput)
{
    auto node   = make_nr_single_input_node();
    auto inputs = llm::input_mappings_t{{"input0", "arg0"}};
    llm::LLMNodeRunner runner{"runner", inputs, node};
    ASSERT_EQ(runner.parent_input_names(), std::vector<std::string>{"input0"});
    ASSERT_EQ(runner.sibling_input_names(), std::vector<std::string>{});
}

TEST_F(TestLLMNodeRunner, SiblingInputExternalNode)
{
    auto node   = make_nr_single_input_node();
    auto inputs = llm::input_mappings_t{{"/sibling", "arg0"}};
    llm::LLMNodeRunner runner{"runner", inputs, node};
    ASSERT_EQ(runner.parent_input_names(), std::vector<std::string>{});
    ASSERT_EQ(runner.sibling_input_names(), std::vector<std::string>{"/sibling"});
}

TEST_F(TestLLMNodeRunner, SiblingInputExternalNodeInputName)
{
    auto node   = make_nr_single_input_node();
    auto inputs = llm::input_mappings_t{{"/sibling/input0", "arg0"}};
    llm::LLMNodeRunner runner{"runner", inputs, node};
    ASSERT_EQ(runner.parent_input_names(), std::vector<std::string>{});
    ASSERT_EQ(runner.sibling_input_names(), std::vector<std::string>{"/sibling/input0"});
}

TEST_F(TestLLMNodeRunner, ParentSiblingInputs)
{
    auto node = std::make_shared<llm::LLMNode>();

    node->add_node("Root1", {{"input0", "arg0"}}, make_nr_single_input_node());
    node->add_node("Root2", {{"input1", "arg0"}}, make_nr_single_input_node());

    auto inputs = llm::input_mappings_t{{"input0", "input0"}, {"/sibling", "input1"}};
    llm::LLMNodeRunner runner{"runner", inputs, node};
    ASSERT_EQ(runner.parent_input_names(), std::vector<std::string>{"input0"});
    ASSERT_EQ(runner.sibling_input_names(), std::vector<std::string>{"/sibling"});
}

TEST_F(TestLLMNodeRunner, DuplicateInput)
{
    auto node   = make_nr_single_input_node();
    auto inputs = llm::input_mappings_t{{"input0", "arg0"}, {"input0", "arg0"}};
    ASSERT_THROW(llm::LLMNodeRunner("runner", inputs, node), std::runtime_error);
}

TEST_F(TestLLMNodeRunner, InvalidExternalNode)
{
    auto node   = make_nr_single_input_node();
    auto inputs = llm::input_mappings_t{{"ext/input0", "arg0"}};
    ASSERT_THROW(llm::LLMNodeRunner("runner", inputs, node), std::invalid_argument);
}

TEST_F(TestLLMNodeRunner, InvalidSingleInputForNode)
{
    auto node   = make_nr_single_input_node();
    auto inputs = llm::input_mappings_t{{"input0", "arg1"}};
    ASSERT_THROW(llm::LLMNodeRunner("runner", inputs, node), std::runtime_error);
}

TEST_F(TestLLMNodeRunner, InvalidMultipleInputsForNode)
{
    auto node   = make_nr_single_input_node();
    auto inputs = llm::input_mappings_t{{"input0", "arg0"}, {"input1", "arg1"}};
    ASSERT_THROW(llm::LLMNodeRunner("runner", inputs, node), std::runtime_error);
}

TEST_F(TestLLMNodeRunner, Execute)
{
    llm::LLMNode node;

    auto runner_1 = node.add_node("Root1", {}, make_nr_dummy_node());
    auto runner_2 = node.add_node("Root2", {{"/Root1"}}, make_nr_single_input_node(), true);

    auto context = std::make_shared<llm::LLMContext>(llm::LLMTask{}, nullptr);

    mrc::coroutines::sync_wait(mrc::coroutines::when_all(runner_1->execute(context), runner_2->execute(context)));

    ASSERT_EQ(context->view_outputs()["Root1"], 0);
    ASSERT_EQ(context->view_outputs()["Root2"], 1);
}
