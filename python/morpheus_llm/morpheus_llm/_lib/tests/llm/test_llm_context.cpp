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

#include "morpheus/_lib/tests/test_utils/common.hpp"  // IWYU pragma: associated
#include "morpheus/messages/control.hpp"              // for ControlMessage
#include "morpheus/utilities/json_types.hpp"          // for PythonByteContainer

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <memory>
#include <stdexcept>
// IWYU pragma: no_include "morpheus_llm/llm/fwd.hpp"

using namespace morpheus;
using namespace morpheus::test;

TEST_CLASS(LLMContext);

TEST_F(TestLLMContext, Initialization)
{
    llm::LLMContext ctx_1;

    nlohmann::json task_dict;
    task_dict = {
        {"task_type", "dictionary"},
        {"model_name", "test"},
    };

    llm::LLMContext ctx_2{llm::LLMTask{"template", task_dict}, nullptr};
    ASSERT_EQ(ctx_2.task().get("task_type"), "dictionary");
    ASSERT_EQ(ctx_2.task().get("model_name"), "test");

    nlohmann::json msg_config;
    msg_config["tasks"] = {{{"type", "llm_engine"}, {"properties", {{"type", "template"}, {"properties", task_dict}}}}};

    auto msg = std::make_shared<ControlMessage>(msg_config);

    llm::LLMContext ctx_3{llm::LLMTask{}, msg};
    ASSERT_EQ(ctx_3.message()->has_task("llm_engine"), true);

    llm::LLMContext ctx_4{llm::LLMTask{"template", task_dict}, msg};
    ASSERT_EQ(ctx_4.message()->has_task("llm_engine"), true);
    ASSERT_EQ(ctx_4.task().get("task_type"), "dictionary");
    ASSERT_EQ(ctx_4.task().get("model_name"), "test");

    auto parent_ctx = std::make_shared<llm::LLMContext>(llm::LLMTask{"template", task_dict}, msg);
    auto inputs     = llm::input_mappings_t{{"/ext1", "input1"}};
    llm::LLMContext ctx_5{parent_ctx, "child", inputs};
    ASSERT_EQ(ctx_5.input_map()[0].external_name, "/ext1");
    ASSERT_EQ(ctx_5.input_map()[0].internal_name, "input1");
    ASSERT_EQ(ctx_5.parent()->message()->has_task("llm_engine"), true);
    ASSERT_EQ(ctx_5.parent()->task().get("task_type"), "dictionary");
    ASSERT_EQ(ctx_5.parent()->task().get("model_name"), "test");
    ASSERT_EQ(ctx_5.name(), "child");
    ASSERT_EQ(ctx_5.full_name(), "/child");
}

TEST_F(TestLLMContext, InitWithLLMTask)
{
    nlohmann::json task_dict;
    task_dict = {
        {"task_type", "dictionary"},
        {"model_name", "test"},
    };

    llm::LLMContext ctx{llm::LLMTask{"template", task_dict}, nullptr};
    ASSERT_EQ(ctx.task().get("task_type"), "dictionary");
    ASSERT_EQ(ctx.task().get("model_name"), "test");
}

TEST_F(TestLLMContext, InitWithControlMessageTask)
{
    nlohmann::json task_dict;
    task_dict = {
        {"task_type", "dictionary"},
        {"model_name", "test"},
    };

    nlohmann::json msg_config;
    msg_config["tasks"] = {{{"type", "llm_engine"}, {"properties", {{"type", "template"}, {"properties", task_dict}}}}};

    auto msg = std::make_shared<ControlMessage>(msg_config);

    llm::LLMContext ctx{llm::LLMTask{}, msg};
    ASSERT_EQ(ctx.message()->has_task("llm_engine"), true);
}

TEST_F(TestLLMContext, InitWithLLMTaskAndControlMessageTask)
{
    nlohmann::json task_dict;
    task_dict = {
        {"task_type", "dictionary"},
        {"model_name", "test"},
    };

    nlohmann::json msg_config;
    msg_config["tasks"] = {{{"type", "llm_engine"}, {"properties", {{"type", "template"}, {"properties", task_dict}}}}};

    auto msg = std::make_shared<ControlMessage>(msg_config);

    llm::LLMContext ctx{llm::LLMTask{"template", task_dict}, msg};
    ASSERT_EQ(ctx.message()->has_task("llm_engine"), true);
    ASSERT_EQ(ctx.task().get("task_type"), "dictionary");
    ASSERT_EQ(ctx.task().get("model_name"), "test");
}

TEST_F(TestLLMContext, InitWithParentContext)
{
    nlohmann::json task_dict;
    task_dict = {
        {"task_type", "dictionary"},
        {"model_name", "test"},
    };

    nlohmann::json msg_config;
    msg_config["tasks"] = {{{"type", "llm_engine"}, {"properties", {{"type", "template"}, {"properties", task_dict}}}}};

    auto msg = std::make_shared<ControlMessage>(msg_config);

    auto parent_ctx = std::make_shared<llm::LLMContext>(llm::LLMTask{"template", task_dict}, msg);
    auto inputs     = llm::input_mappings_t{{"/ext1", "input1"}};
    llm::LLMContext ctx{parent_ctx, "child", inputs};
    ASSERT_EQ(ctx.input_map()[0].external_name, "/ext1");
    ASSERT_EQ(ctx.input_map()[0].internal_name, "input1");
    ASSERT_EQ(ctx.parent()->message()->has_task("llm_engine"), true);
    ASSERT_EQ(ctx.parent()->task().get("task_type"), "dictionary");
    ASSERT_EQ(ctx.parent()->task().get("model_name"), "test");
    ASSERT_EQ(ctx.name(), "child");
    ASSERT_EQ(ctx.full_name(), "/child");
}

TEST_F(TestLLMContext, SetOutput)
{
    llm::LLMContext ctx{llm::LLMTask{}, nullptr};
    nlohmann::json outputs;
    outputs = {{"key1", "val1"}, {"key2", "val2"}};
    ctx.set_output(outputs);

    const auto& json_outputs = ctx.all_outputs();
    ASSERT_EQ(json_outputs.size(), 2);
    ASSERT_EQ(json_outputs["key1"], "val1");
    ASSERT_EQ(json_outputs["key2"], "val2");
}

TEST_F(TestLLMContext, SetOutputDict)
{
    llm::LLMContext ctx{llm::LLMTask{}, nullptr};
    nlohmann::json outputs;
    outputs = {{"key1", "val1"}, {"key2", "val2"}};

    ctx.set_output("output", outputs);

    const auto& json_outputs = ctx.all_outputs();
    ASSERT_EQ(json_outputs["output"]["key1"], "val1");
    ASSERT_EQ(json_outputs["output"]["key2"], "val2");
}

TEST_F(TestLLMContext, PushPop)
{
    auto parent_ctx = std::make_shared<llm::LLMContext>(llm::LLMTask{}, nullptr);
    auto inputs     = llm::input_mappings_t{{"/ext1", "input1"}};
    auto child_ctx  = parent_ctx->push("child", inputs);

    ASSERT_EQ(child_ctx->name(), "child");
    ASSERT_EQ(child_ctx->input_map()[0].external_name, "/ext1");
    ASSERT_EQ(child_ctx->input_map()[0].internal_name, "input1");

    nlohmann::json outputs;
    outputs = {{"key1", "val1"}, {"key2", "val2"}};
    child_ctx->set_output(outputs);

    const auto& child_json_outputs = child_ctx->all_outputs();
    ASSERT_EQ(child_json_outputs["key1"], "val1");
    ASSERT_EQ(child_json_outputs["key2"], "val2");

    child_ctx->pop();
    const auto& parent_json_outputs = child_ctx->parent()->all_outputs();
    ASSERT_EQ(parent_json_outputs["child"]["key1"], "val1");
    ASSERT_EQ(parent_json_outputs["child"]["key2"], "val2");
}

TEST_F(TestLLMContext, PopWithoutPush)
{
    auto parent_ctx = std::make_shared<llm::LLMContext>(llm::LLMTask{}, nullptr);
    auto inputs     = llm::input_mappings_t{{"/ext1", "input1"}};
    llm::LLMContext child_ctx{parent_ctx, "child", inputs};

    ASSERT_EQ(child_ctx.name(), "child");
    ASSERT_EQ(child_ctx.input_map()[0].external_name, "/ext1");
    ASSERT_EQ(child_ctx.input_map()[0].internal_name, "input1");

    nlohmann::json outputs;
    outputs = {{"key1", "val1"}, {"key2", "val2"}};
    child_ctx.set_output(outputs);

    const auto& child_json_outputs = child_ctx.all_outputs();
    ASSERT_EQ(child_json_outputs["key1"], "val1");
    ASSERT_EQ(child_json_outputs["key2"], "val2");

    child_ctx.pop();

    const auto& parent_json_outputs = parent_ctx->all_outputs();
    ASSERT_EQ(parent_json_outputs["child"]["key1"], "val1");
    ASSERT_EQ(parent_json_outputs["child"]["key2"], "val2");
}

TEST_F(TestLLMContext, PopSelectOneOutput)
{
    auto parent_ctx = std::make_shared<llm::LLMContext>(llm::LLMTask{}, nullptr);
    auto inputs     = llm::input_mappings_t{{"/ext1", "input1"}};
    auto child_ctx  = parent_ctx->push("child", inputs);

    nlohmann::json outputs;
    outputs = {{"key1", "val1"}, {"key2", "val2"}, {"key3", "val3"}};
    child_ctx->set_output(outputs);

    const auto& child_json_outputs = child_ctx->all_outputs();
    ASSERT_EQ(child_json_outputs["key1"], "val1");
    ASSERT_EQ(child_json_outputs["key2"], "val2");
    ASSERT_EQ(child_json_outputs["key3"], "val3");

    child_ctx->set_output_names({"key2"});
    child_ctx->pop();

    const auto& parent_json_outputs = parent_ctx->all_outputs();
    ASSERT_EQ(parent_json_outputs["child"].size(), 1);
    ASSERT_EQ(parent_json_outputs["child"], "val2");
}

TEST_F(TestLLMContext, PopSelectMultipleOutputs)
{
    auto parent_ctx = std::make_shared<llm::LLMContext>(llm::LLMTask{}, nullptr);
    auto inputs     = llm::input_mappings_t{{"/ext1", "input1"}};
    auto child_ctx  = parent_ctx->push("child", inputs);

    nlohmann::json outputs;
    outputs = {{"key1", "val1"}, {"key2", "val2"}, {"key3", "val3"}};
    child_ctx->set_output(outputs);

    const auto& child_json_outputs = child_ctx->all_outputs();
    ASSERT_EQ(child_json_outputs["key1"], "val1");
    ASSERT_EQ(child_json_outputs["key2"], "val2");
    ASSERT_EQ(child_json_outputs["key3"], "val3");

    child_ctx->set_output_names({"key2", "key3"});
    child_ctx->pop();

    const auto& parent_json_outputs = child_ctx->parent()->all_outputs();
    ASSERT_EQ(parent_json_outputs["child"].size(), 2);
    ASSERT_EQ(parent_json_outputs["child"]["key2"], "val2");
    ASSERT_EQ(parent_json_outputs["child"]["key3"], "val3");
}

TEST_F(TestLLMContext, SingleInputMappingValid)
{
    auto parent_ctx = std::make_shared<llm::LLMContext>(llm::LLMTask{}, nullptr);
    nlohmann::json outputs;
    outputs = {{"parent_out", "val1"}};
    parent_ctx->set_output(outputs);

    auto inputs = llm::input_mappings_t{{"/parent_out", "input1"}};
    llm::LLMContext child_ctx{parent_ctx, "child", inputs};
    ASSERT_EQ(child_ctx.get_input(), "val1");
    ASSERT_EQ(child_ctx.get_input("input1"), "val1");
    ASSERT_EQ(child_ctx.get_inputs()["input1"], "val1");
    ASSERT_THROW(child_ctx.get_input("input2"), std::runtime_error);
}

TEST_F(TestLLMContext, SingleInputMappingInvalid)
{
    auto parent_ctx = std::make_shared<llm::LLMContext>(llm::LLMTask{}, nullptr);
    nlohmann::json outputs;
    outputs = {{"parent_out", "val1"}};
    parent_ctx->set_output(outputs);

    auto inputs = llm::input_mappings_t{{"/invalid", "input1"}};
    llm::LLMContext child_ctx{parent_ctx, "child", inputs};
    ASSERT_THROW(child_ctx.get_input(), std::runtime_error);
    ASSERT_THROW(child_ctx.get_input("input1"), std::runtime_error);
    ASSERT_THROW(child_ctx.get_inputs()["input1"], std::runtime_error);
}

TEST_F(TestLLMContext, MultipleInputMappingsValid)
{
    auto parent_ctx = std::make_shared<llm::LLMContext>(llm::LLMTask{}, nullptr);
    nlohmann::json outputs;
    outputs = {{"parent_out1", "val1"}, {"parent_out2", "val2"}};
    parent_ctx->set_output(outputs);

    auto inputs = llm::input_mappings_t{{"/parent_out1", "input1"}, {"/parent_out2", "input2"}};
    llm::LLMContext child_ctx{parent_ctx, "child", inputs};
    ASSERT_EQ(child_ctx.get_input("input1"), "val1");
    ASSERT_EQ(child_ctx.get_input("input2"), "val2");

    auto child_inputs = child_ctx.get_inputs();

    ASSERT_EQ(child_inputs["input1"], "val1");
    ASSERT_EQ(child_inputs["input2"], "val2");
    ASSERT_THROW(child_ctx.get_input(), std::runtime_error);
    ASSERT_THROW(child_ctx.get_input("input3"), std::runtime_error);
}

TEST_F(TestLLMContext, MultipleInputMappingsSingleInvalid)
{
    auto parent_ctx = std::make_shared<llm::LLMContext>(llm::LLMTask{}, nullptr);
    nlohmann::json outputs;
    outputs = {{"parent_out1", "val1"}, {"parent_out2", "val2"}};
    parent_ctx->set_output(outputs);

    auto inputs = llm::input_mappings_t{{"/parent_out1", "input1"}, {"/invalid", "input2"}};
    llm::LLMContext child_ctx{parent_ctx, "child", inputs};
    ASSERT_EQ(child_ctx.get_input("input1"), "val1");
    ASSERT_THROW(child_ctx.get_input("input2"), std::runtime_error);
    ASSERT_THROW(child_ctx.get_inputs(), std::runtime_error);
}

TEST_F(TestLLMContext, MultipleInputMappingsBothInvalid)
{
    auto parent_ctx = std::make_shared<llm::LLMContext>(llm::LLMTask{}, nullptr);
    nlohmann::json outputs;
    outputs = {{"parent_out1", "val1"}, {"parent_out2", "val2"}};
    parent_ctx->set_output(outputs);

    auto inputs = llm::input_mappings_t{{"/invalid1", "input1"}, {"/invalid2", "input2"}};
    llm::LLMContext child_ctx{parent_ctx, "child", inputs};
    ASSERT_THROW(child_ctx.get_input("input1"), std::runtime_error);
    ASSERT_THROW(child_ctx.get_input("input2"), std::runtime_error);
    ASSERT_THROW(child_ctx.get_inputs(), std::runtime_error);
}
