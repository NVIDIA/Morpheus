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
#include "morpheus_llm/llm/llm_task.hpp"
#include "morpheus_llm/llm/llm_task_handler.hpp"
#include "morpheus_llm/llm/llm_task_handler_runner.hpp"

#include "morpheus/_lib/tests/test_utils/common.hpp"  // IWYU pragma: associated
#include "morpheus/messages/control.hpp"              // for ControlMessage
#include "morpheus/types.hpp"
#include "morpheus/utilities/json_types.hpp"  // for PythonByteContainer

#include <gtest/gtest.h>
#include <mrc/channel/forward.hpp>  // IWYU pragma: keep
#include <mrc/coroutines/sync_wait.hpp>
#include <mrc/coroutines/task.hpp>  // IWYU pragma: keep
#include <nlohmann/json.hpp>        // for nlohmann::json

#include <algorithm>  // for max (implied usage)
#include <coroutine>
#include <memory>
#include <optional>  // for optional
#include <string>
// IWYU pragma: no_include "morpheus_llm/llm/fwd.hpp"
// IWYU pragma: no_include <pymrc/utilities/object_wrappers.hpp>

using namespace morpheus;
using namespace morpheus::test;
using namespace mrc;

class TestTaskHandler : public llm::LLMTaskHandler
{
  public:
    TestTaskHandler() = default;

    TestTaskHandler(std::vector<std::string> input_names)
    {
        m_input_names = input_names;
    }

    ~TestTaskHandler() override = default;

    std::vector<std::string> get_input_names() const override
    {
        return m_input_names;
    }

    Task<return_t> try_handle(std::shared_ptr<llm::LLMContext> context) override
    {
        std::vector<std::shared_ptr<ControlMessage>> results;

        for (auto& name : m_input_names)
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
    std::vector<std::string> m_input_names;
};

TEST_CLASS(LLMTaskHandlerRunner);

TEST_F(TestLLMTaskHandlerRunner, Initialization)
{
    auto names   = std::vector<std::string>{"input0", "input1"};
    auto handler = std::make_shared<TestTaskHandler>(names);
    auto inputs  = llm::input_mappings_t{{"/ext0", "input0"}, {"/ext1", "input1"}};

    llm::LLMTaskHandlerRunner runner{inputs, handler};

    auto returned_mappings = runner.input_names();
    ASSERT_EQ(returned_mappings.size(), 2);
    ASSERT_EQ(returned_mappings[0].external_name, "/ext0");
    ASSERT_EQ(returned_mappings[0].internal_name, "input0");
    ASSERT_EQ(returned_mappings[1].external_name, "/ext1");
    ASSERT_EQ(returned_mappings[1].internal_name, "input1");
}

TEST_F(TestLLMTaskHandlerRunner, TryHandle)
{
    auto names   = std::vector<std::string>{"input0", "input1"};
    auto handler = std::make_shared<TestTaskHandler>(names);
    auto inputs  = llm::input_mappings_t{{"/ext0", "input0"}, {"/ext1", "input1"}};

    llm::LLMTaskHandlerRunner runner{inputs, handler};

    auto context  = std::make_shared<llm::LLMContext>(llm::LLMTask{}, nullptr);
    auto out_msgs = coroutines::sync_wait(runner.try_handle(context));

    ASSERT_EQ(out_msgs->size(), 2);
    ASSERT_EQ(out_msgs->at(0)->get_tasks().size(), 1);
    ASSERT_EQ(out_msgs->at(0)->get_tasks()["template"][0]["task_type"], "dictionary");
    ASSERT_EQ(out_msgs->at(0)->get_tasks()["template"][0]["model_name"], "test");
    ASSERT_EQ(out_msgs->at(0)->get_tasks()["template"][0]["input"], "input0");
    ASSERT_EQ(out_msgs->at(1)->get_tasks().size(), 1);
    ASSERT_EQ(out_msgs->at(1)->get_tasks()["template"][0]["task_type"], "dictionary");
    ASSERT_EQ(out_msgs->at(1)->get_tasks()["template"][0]["model_name"], "test");
    ASSERT_EQ(out_msgs->at(1)->get_tasks()["template"][0]["input"], "input1");
}
