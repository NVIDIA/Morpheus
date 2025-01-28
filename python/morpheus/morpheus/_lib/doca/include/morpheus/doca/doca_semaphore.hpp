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

#include <doca_gpunetio.h>

#include <cstdint>
#include <memory>

namespace morpheus::doca {

/**
 * @brief Creates and manages the lifetime of a GPUNetIO Semaphore.
 *
 * GPUNetIO Semaphores are used in a round-robin fashion to create locks over regions of Receive
 * Queue memory so they can be read in device kernels.
 */
struct DocaSemaphore
{
  private:
    std::shared_ptr<DocaContext> m_context;
    uint16_t m_size;
    doca_gpu_semaphore* m_semaphore;
    doca_gpu_semaphore_gpu* m_semaphore_gpu;

  public:
    DocaSemaphore(std::shared_ptr<DocaContext> context, uint16_t size);
    ~DocaSemaphore();

    doca_gpu_semaphore_gpu* gpu_ptr();
    uint16_t size();
    void* get_info_cpu(uint32_t idx);
    bool is_ready(uint32_t idx);
    void set_free(uint32_t idx);
};

}  // namespace morpheus::doca
