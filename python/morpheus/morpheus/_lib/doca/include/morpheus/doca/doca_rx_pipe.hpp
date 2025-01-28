/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/doca/common.hpp"
#include "morpheus/doca/doca_context.hpp"
#include "morpheus/doca/doca_rx_queue.hpp"

#include <doca_flow.h>

#include <memory>
#include <vector>

namespace morpheus::doca {

/**
 * @brief Creates and manages the lifetime of a GPUNetIO Receive Pipe for a given GPUNetIO Receive Queue.
 *
 * Pipes are used to filter and/or forward packets to other Pipes or Receive Queues. A Root Pipe
 * is the primary Pipe where packets come in, and can then be forwarded to other Pipes. This is how
 * TCP/UDP as well as other types of filtering is done with GPUNetIO. Eventually packets will be
 * placed in a Receive Queue at which point they can be read using a Semaphore.
 *
 * In this implementation, a single Root Pipe is connected to a single TCP-filtering Pipe which is
 * then connected to the given Receive Queue.
 */
struct DocaRxPipe
{
  private:
    std::shared_ptr<DocaContext> m_context;
    std::vector<std::shared_ptr<morpheus::doca::DocaRxQueue>> m_rxq;
    enum doca_traffic_type m_traffic_type;
    doca_flow_pipe* m_pipe;
    doca_flow_pipe* m_root_pipe;

  public:
    DocaRxPipe(std::shared_ptr<DocaContext> context,
               std::vector<std::shared_ptr<morpheus::doca::DocaRxQueue>> rxq,
               doca_traffic_type const type);
    ~DocaRxPipe();
};

}  // namespace morpheus::doca
