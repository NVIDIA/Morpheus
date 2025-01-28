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

#include "morpheus/doca/rte_context.hpp"

#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_flow_crypto.h>
#include <doca_log.h>

#include <cstdint>
#include <memory>
#include <string>

#define GPU_PAGE_SIZE (1UL << 16)

namespace morpheus::doca {

/**
 * @brief Manages the lifetime of DOCA as it relates to GPUNetIO
 */
struct DocaContext
{
  private:
    doca_gpu* m_gpu;
    doca_dev* m_dev;
    doca_flow_port* m_flow_port;
    uint16_t m_nic_port;
    uint32_t m_max_queue_count;
    std::unique_ptr<RTEContext> m_rte_context;
    doca_log_backend* m_sdk_log;

  public:
    DocaContext(std::string nic_addr, std::string gpu_addr);
    ~DocaContext();

    doca_gpu* gpu();
    doca_dev* dev();
    uint16_t nic_port();
    doca_flow_port* flow_port();
};

}  // namespace morpheus::doca
