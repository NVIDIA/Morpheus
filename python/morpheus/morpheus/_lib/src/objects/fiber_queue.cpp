/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/objects/fiber_queue.hpp"

#include <boost/fiber/channel_op_status.hpp>
#include <glog/logging.h>  // for LOG, FATAL
#include <pybind11/gil.h>  // for gil_scoped_release
#include <pybind11/pybind11.h>

#include <chrono>
#include <functional>  // for ref, reference_wrapper
#include <memory>
#include <ostream>    // needed by GLOG
#include <ratio>      // for ratio needed for std::chrono::duration
#include <stdexcept>  // for invalid_argument, runtime_error
#include <utility>

namespace morpheus {
/****** Component public implementations *******************/
/****** FiberQueue****************************************/
FiberQueue::FiberQueue(size_t max_size) : m_queue(max_size) {}

boost::fibers::channel_op_status FiberQueue::put(pybind11::object&& item, bool block, float timeout)
{
    if (!block)
    {
        return m_queue.try_push(std::move(item));
    }
    else if (timeout > 0.0)
    {
        return m_queue.push_wait_for(
            std::move(item), std::chrono::duration_cast<std::chrono::seconds>(std::chrono::duration<float>(timeout)));
    }
    else
    {
        // Blocking no timeout
        return m_queue.push(std::move(item));
    }
}

boost::fibers::channel_op_status FiberQueue::get(pybind11::object& item, bool block, float timeout)
{
    if (!block)
    {
        return m_queue.try_pop(std::ref(item));
    }
    else if (timeout > 0.0)
    {
        return m_queue.pop_wait_for(
            std::ref(item), std::chrono::duration_cast<std::chrono::seconds>(std::chrono::duration<float>(timeout)));
    }
    else
    {
        // Blocking no timeout
        return m_queue.pop(std::ref(item));
    }
}

void FiberQueue::close()
{
    m_queue.close();
}

bool FiberQueue::is_closed()
{
    return m_queue.is_closed();
}

void FiberQueue::join()
{
    // TODO(MDD): Not sure how to join a buffered channel
}

/****** FiberQueueInterfaceProxy *************************/
std::shared_ptr<morpheus::FiberQueue> FiberQueueInterfaceProxy::init(std::size_t max_size)
{
    if (max_size < 2 || ((max_size & (max_size - 1)) != 0))
    {
        throw std::invalid_argument("max_size must be greater than 1 and a power of 2.");
    }

    // Create a new shared_ptr
    return std::make_shared<morpheus::FiberQueue>(max_size);
}

void FiberQueueInterfaceProxy::put(morpheus::FiberQueue& self, pybind11::object item, bool block, float timeout)
{
    boost::fibers::channel_op_status status;

    // Release the GIL and try to move it
    {
        pybind11::gil_scoped_release nogil;

        status = self.put(std::move(item), block, timeout);
    }

    switch (status)
    {
    case boost::fibers::channel_op_status::success:
        return;
    case boost::fibers::channel_op_status::empty: {
        LOG(FATAL) << "FiberQueue::put should never return empty.";
    }
    case boost::fibers::channel_op_status::full:
    case boost::fibers::channel_op_status::timeout: {
        // Raise queue.Full
        pybind11::object exc_class = pybind11::module_::import("queue").attr("Full");

        PyErr_SetNone(exc_class.ptr());

        throw pybind11::error_already_set();
    }
    case boost::fibers::channel_op_status::closed: {
        // Raise queue.Full
        pybind11::object exc_class = pybind11::module_::import("morpheus.utils.producer_consumer_queue").attr("Closed");

        PyErr_SetNone(exc_class.ptr());

        throw pybind11::error_already_set();
    }
    }
}

pybind11::object FiberQueueInterfaceProxy::get(morpheus::FiberQueue& self, bool block, float timeout)
{
    boost::fibers::channel_op_status status;

    pybind11::object item;

    // Release the GIL and try to move it
    {
        pybind11::gil_scoped_release nogil;

        status = self.get(std::ref(item), block, timeout);
    }

    switch (status)
    {
    case boost::fibers::channel_op_status::success:
        return item;
    case boost::fibers::channel_op_status::empty: {
        // Raise queue.Empty
        pybind11::object exc_class = pybind11::module_::import("queue").attr("Empty");

        PyErr_SetNone(exc_class.ptr());

        throw pybind11::error_already_set();
    }
    case boost::fibers::channel_op_status::full:
    case boost::fibers::channel_op_status::timeout: {
        // Raise queue.Full
        pybind11::object exc_class = pybind11::module_::import("queue").attr("Empty");

        PyErr_SetNone(exc_class.ptr());

        throw pybind11::error_already_set();
    }
    case boost::fibers::channel_op_status::closed: {
        // Raise queue.Full
        pybind11::object exc_class = pybind11::module_::import("morpheus.utils.producer_consumer_queue").attr("Closed");

        PyErr_SetNone(exc_class.ptr());

        throw pybind11::error_already_set();
    }
    default:
        throw std::runtime_error("Unknown channel status");
    }
}

void FiberQueueInterfaceProxy::close(morpheus::FiberQueue& self)
{
    self.close();
}

bool FiberQueueInterfaceProxy::is_closed(morpheus::FiberQueue& self)
{
    return self.is_closed();
}

morpheus::FiberQueue& FiberQueueInterfaceProxy::enter(morpheus::FiberQueue& self)
{
    return self;
}

void FiberQueueInterfaceProxy::exit(morpheus::FiberQueue& self,
                                    const pybind11::object& type,
                                    const pybind11::object& value,
                                    const pybind11::object& traceback)
{
    self.close();
}

}  // namespace morpheus
