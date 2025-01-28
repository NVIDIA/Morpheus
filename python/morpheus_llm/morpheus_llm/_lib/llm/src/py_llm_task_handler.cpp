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

#include "py_llm_task_handler.hpp"

#include "morpheus_llm/llm/llm_context.hpp"  // IWYU pragma: keep

#include "morpheus/messages/control.hpp"  // IWYU pragma: keep

#include <glog/logging.h>
#include <mrc/coroutines/task.hpp>  // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // IWYU pragma: keep
#include <pymrc/coro.hpp>  // IWYU pragma: keep

#include <coroutine>

namespace morpheus::llm {
namespace py = pybind11;

PyLLMTaskHandler::~PyLLMTaskHandler() = default;

std::vector<std::string> PyLLMTaskHandler::get_input_names() const
{
    PYBIND11_OVERRIDE_PURE(std::vector<std::string>, LLMTaskHandler, get_input_names);
}

Task<LLMTaskHandler::return_t> PyLLMTaskHandler::try_handle(std::shared_ptr<LLMContext> context)
{
    MRC_PYBIND11_OVERRIDE_CORO_PURE(LLMTaskHandler::return_t, LLMTaskHandler, try_handle, context);
}

}  // namespace morpheus::llm
