/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/llm/input_map.hpp"
#include "morpheus/llm/llm_context.hpp"
#include "morpheus/llm/llm_lambda_node.hpp"
#include "morpheus/llm/llm_node.hpp"
#include "morpheus/llm/llm_node_runner.hpp"
#include "morpheus/llm/llm_task.hpp"
#include "morpheus/llm/llm_task_handler.hpp"
#include "morpheus/llm/llm_task_handler_runner.hpp"
#include "morpheus/types.hpp"

#include <gtest/gtest.h>
#include <mrc/channel/forward.hpp>
#include <mrc/coroutines/sync_wait.hpp>
#include <mrc/coroutines/task.hpp>

#include <coroutine>
#include <memory>
#include <stdexcept>
#include <string>

using namespace morpheus;
using namespace morpheus::test;
using namespace mrc;

class TestTaskHandler : public llm::LLMTaskHandler
{
  public:
    TestTaskHandler() = default;

    TestTaskHandler(std::vector<std::string> input_names)
    {
        _input_names = input_names;
    }

    ~TestTaskHandler() = default;

    std::vector<std::string> get_input_names() const
    {
        return _input_names;
    }

    Task<return_t> try_handle(std::shared_ptr<llm::LLMContext> context)
    {
        std::vector<std::shared_ptr<ControlMessage>> results;

        for (auto& name : _input_names)
        {
            auto msg_config = nlohmann::json();
            nlohmann::json task_properties;
            task_properties          = {{"task_type", "dictionary"}, {"model_name", "test"}};
            task_properties["input"] = name;
            msg_config["tasks"]      = {{{"type", "template"}, {"properties", task_properties}}};

            auto msg = std::make_shared<ControlMessage>(msg_config);

            results.push_back(msg);
        }

        co_return results;
    }

  private:
    std::vector<std::string> _input_names;
};

TEST_CLASS(LLMTaskHandlerRunner);

TEST_F(TestLLMTaskHandlerRunner, TryHandle)
{
    auto names = std::vector<std::string>{"input1", "input2"};
    TestTaskHandler handler{names};

    ASSERT_EQ(handler.get_input_names(), names);

    auto context = std::make_shared<llm::LLMContext>(llm::LLMTask{}, nullptr);

    auto out_msgs = coroutines::sync_wait(handler.try_handle(context));

    ASSERT_EQ(0, 0);
    ASSERT_EQ(out_msgs->size(), 2);
    ASSERT_EQ(out_msgs->at(0)->get_tasks().size(), 1);
    ASSERT_EQ(out_msgs->at(0)->get_tasks()["template"][0]["task_type"], "dictionary");
    ASSERT_EQ(out_msgs->at(0)->get_tasks()["template"][0]["model_name"], "test");
    ASSERT_EQ(out_msgs->at(0)->get_tasks()["template"][0]["input"], "input1");
    ASSERT_EQ(out_msgs->at(1)->get_tasks().size(), 1);
    ASSERT_EQ(out_msgs->at(1)->get_tasks()["template"][0]["task_type"], "dictionary");
    ASSERT_EQ(out_msgs->at(1)->get_tasks()["template"][0]["model_name"], "test");
    ASSERT_EQ(out_msgs->at(1)->get_tasks()["template"][0]["input"], "input2");
}
