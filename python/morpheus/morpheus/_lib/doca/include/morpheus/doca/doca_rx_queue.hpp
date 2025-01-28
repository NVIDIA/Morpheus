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

#include "morpheus/doca/doca_context.hpp"
#include "morpheus/doca/doca_mem.hpp"

#include <doca_ctx.h>
#include <doca_eth_rxq.h>

#include <memory>

namespace morpheus::doca {

/**
 * @brief Creates and manages the lifetime of a GPUNetIO Receive Queue.
 *
 * A Receive Queue is used to buffer packets received from a GPUNetIO Pipe.
 */
struct DocaRxQueue
{
  private:
    std::shared_ptr<DocaContext> m_context;
    doca_gpu_eth_rxq* m_rxq_info_gpu;
    doca_eth_rxq* m_rxq_info_cpu;
    doca_mmap* m_packet_mmap;
    doca_ctx* m_doca_ctx;
    std::unique_ptr<DocaMem<void>> m_packet_memory;

  public:
    DocaRxQueue(std::shared_ptr<DocaContext> context);
    ~DocaRxQueue();

    doca_gpu_eth_rxq* rxq_info_gpu();
    doca_eth_rxq* rxq_info_cpu();
};

}  // namespace morpheus::doca
