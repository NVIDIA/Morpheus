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

#pragma once

#include "morpheus/export.h"

#include <boost/fiber/buffered_channel.hpp>
#include <boost/fiber/channel_op_status.hpp>
#include <pybind11/pybind11.h>  // IWYU pragma: keep
#include <pybind11/pytypes.h>

#include <cstddef>
#include <memory>

namespace morpheus {
/****** Component public implementations *******************/
/****** FiberQueue****************************************/

/**
 * @addtogroup objects
 * @{
 * @file
 */

/**
 * @brief This class acts as a collection or linear data structure that stores elements in FIFO (First In, First Out)
 * order
 *
 */
class MORPHEUS_EXPORT FiberQueue
{
  public:
    FiberQueue(std::size_t max_size);

    /**
     * @brief Item to the queue. Await the acknowledgement delays based on the timeout that has been specified.
     *
     * @param item
     * @param block
     * @param timeout
     * @return boost::fibers::channel_op_status
     */
    boost::fibers::channel_op_status put(pybind11::object&& item, bool block = true, float timeout = 0.0);

    /**
     * @brief Retrieves item from head of the queue.
     *
     * @param item
     * @param block
     * @param timeout
     * @return boost::fibers::channel_op_status
     */
    boost::fibers::channel_op_status get(pybind11::object& item, bool block = true, float timeout = 0.0);

    /**
     * TODO(Documentation)
     */
    void close();

    /**
     * TODO(Documentation)
     */
    bool is_closed();

    /**
     * TODO(Documentation)
     */
    void join();

  private:
    boost::fibers::buffered_channel<pybind11::object> m_queue;
};

/****** FiberQueueInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MORPHEUS_EXPORT FiberQueueInterfaceProxy
{
    /**
     * @brief Create and initialize a FIberQueue, and return a shared pointer to the result
     *
     * @param max_size
     * @return std::shared_ptr<morpheus::FiberQueue>
     */
    static std::shared_ptr<morpheus::FiberQueue> init(std::size_t max_size);

    /**
     * TODO(Documentation)
     */
    static void put(morpheus::FiberQueue& self, pybind11::object item, bool block = true, float timeout = 0.0);

    /**
     * TODO(Documentation)
     */
    static pybind11::object get(morpheus::FiberQueue& self, bool block = true, float timeout = 0.0);

    /**
     * TODO(Documentation)
     */
    static void close(morpheus::FiberQueue& self);

    static bool is_closed(morpheus::FiberQueue& self);

    // Context manager methods
    static morpheus::FiberQueue& enter(morpheus::FiberQueue& self);
    static void exit(morpheus::FiberQueue& self,
                     const pybind11::object& type,
                     const pybind11::object& value,
                     const pybind11::object& traceback);
};

/** @} */  // end of group
}  // namespace morpheus
