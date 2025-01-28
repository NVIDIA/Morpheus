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

#pragma once
#include "morpheus_llm/llm/input_map.hpp"
#include "morpheus_llm/llm/llm_engine.hpp"
#include "morpheus_llm/llm/llm_task_handler.hpp"
#include "py_llm_node.hpp"

#include "morpheus/export.h"
#include "morpheus/messages/control.hpp"
#include "morpheus/types.hpp"

#include <boost/fiber/fiber.hpp>
#include <boost/fiber/future/async.hpp>
#include <boost/fiber/future/future.hpp>
#include <boost/fiber/future/future_status.hpp>
#include <boost/fiber/future/promise.hpp>
#include <boost/fiber/policy.hpp>
#include <glog/logging.h>
#include <mrc/channel/status.hpp>
#include <mrc/coroutines/async_generator.hpp>
#include <mrc/coroutines/closable_ring_buffer.hpp>
#include <mrc/coroutines/concepts/awaitable.hpp>
#include <mrc/coroutines/detail/void_value.hpp>
#include <mrc/coroutines/schedule_on.hpp>
#include <mrc/coroutines/scheduler.hpp>
#include <mrc/coroutines/task.hpp>
#include <mrc/coroutines/task_container.hpp>
#include <mrc/node/sink_channel_owner.hpp>
#include <mrc/node/sink_properties.hpp>
#include <mrc/node/source_channel_owner.hpp>
#include <mrc/node/source_properties.hpp>
#include <mrc/runnable/context.hpp>
#include <mrc/runnable/forward.hpp>
#include <mrc/runnable/runnable.hpp>
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pymrc/asyncio_runnable.hpp>
#include <pymrc/coro.hpp>
#include <pymrc/types.hpp>
#include <pymrc/utilities/acquire_gil.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <coroutine>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <ratio>
#include <stdexcept>
#include <stop_token>
#include <utility>

namespace morpheus::llm {
namespace py = pybind11;

class MORPHEUS_EXPORT PyLLMEngineStage
  : public mrc::pymrc::AsyncioRunnable<std::shared_ptr<ControlMessage>, std::shared_ptr<ControlMessage>>
{
  public:
    PyLLMEngineStage(std::shared_ptr<LLMEngine> engine) : m_engine(std::move(engine)) {}

    ~PyLLMEngineStage() override = default;

    static std::shared_ptr<mrc::segment::Object<PyLLMEngineStage>> init(mrc::segment::Builder& builder,
                                                                        const std::string& name,
                                                                        std::shared_ptr<LLMEngine> engine)
    {
        auto stage = builder.construct_object<PyLLMEngineStage>(name, std::move(engine));

        return stage;
    }

  private:
    mrc::coroutines::AsyncGenerator<std::shared_ptr<ControlMessage>> on_data(
        std::shared_ptr<ControlMessage>&& data, std::shared_ptr<mrc::coroutines::Scheduler> on) override
    {
        auto result = co_await m_engine->run(std::move(data));

        // Push the output messages
        for (auto&& out_message : result)
        {
            co_yield std::move(out_message);
        }

        co_return;
    }

    std::shared_ptr<LLMEngine> m_engine;
};

}  // namespace morpheus::llm
