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
#include "morpheus/doca/error.hpp"

#include <doca_gpunetio.h>

#include <memory>

namespace morpheus::doca {

/**
 * @brief A piece of memory aligned to GPU page size managed by DOCA, optionally addressable from both host and device.
 */
template <typename T>
struct DocaMem
{
  private:
    std::shared_ptr<DocaContext> m_context;
    T* m_mem_gpu;
    T* m_mem_cpu;

  public:
    DocaMem(std::shared_ptr<DocaContext> context, size_t count, doca_gpu_mem_type mem_type);
    ~DocaMem();

    T* gpu_ptr();
    T* cpu_ptr();
};

template <typename T>
DocaMem<T>::DocaMem(std::shared_ptr<morpheus::doca::DocaContext> context, size_t count, doca_gpu_mem_type mem_type) :
  m_context(context)
{
    DOCA_TRY(doca_gpu_mem_alloc(context->gpu(),
                                count * sizeof(std::conditional_t<std::is_void_v<T>, uint8_t, T>),
                                GPU_PAGE_SIZE,
                                mem_type,
                                (void**)(&m_mem_gpu),
                                (void**)(&m_mem_cpu)));
}

template <typename T>
DocaMem<T>::~DocaMem()
{
    DOCA_TRY(doca_gpu_mem_free(m_context->gpu(), m_mem_gpu));
}

template <typename T>
T* DocaMem<T>::gpu_ptr()
{
    return m_mem_gpu;
}

template <typename T>
T* DocaMem<T>::cpu_ptr()
{
    return m_mem_cpu;
}

}  // namespace morpheus::doca
