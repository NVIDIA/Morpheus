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

#include "morpheus/doca/doca_rx_queue.hpp"

#include "morpheus/doca/common.hpp"
#include "morpheus/doca/error.hpp"
#include "morpheus/utilities/error.hpp"

#include <doca_error.h>
#include <doca_mmap.h>
#include <doca_types.h>
#include <glog/logging.h>

#include <cstdint>
#include <ostream>

namespace morpheus::doca {

DocaRxQueue::DocaRxQueue(std::shared_ptr<DocaContext> context) :
  m_context(context),
  m_rxq_info_gpu(nullptr),
  m_rxq_info_cpu(nullptr),
  m_packet_mmap(nullptr),
  m_doca_ctx(nullptr)
{
    uint32_t cyclic_buffer_size;
    DOCA_TRY(doca_eth_rxq_create(context->dev(), MAX_PKT_NUM, MAX_PKT_SIZE, &(m_rxq_info_cpu)));
    DOCA_TRY(doca_eth_rxq_set_type(m_rxq_info_cpu, DOCA_ETH_RXQ_TYPE_CYCLIC));
    DOCA_TRY(doca_eth_rxq_estimate_packet_buf_size(
        DOCA_ETH_RXQ_TYPE_CYCLIC, 0, 0, MAX_PKT_SIZE, MAX_PKT_NUM, 0, &cyclic_buffer_size));
    DOCA_TRY(doca_mmap_create(&m_packet_mmap));
    DOCA_TRY(doca_mmap_add_dev(m_packet_mmap, context->dev()));

    m_packet_memory = std::make_unique<DocaMem<void>>(m_context, cyclic_buffer_size, DOCA_GPU_MEM_TYPE_GPU);

    DOCA_TRY(doca_mmap_set_memrange(m_packet_mmap, m_packet_memory->gpu_ptr(), cyclic_buffer_size));
    DOCA_TRY(doca_mmap_set_permissions(m_packet_mmap,
                                       DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_PCI_RELAXED_ORDERING));
    DOCA_TRY(doca_mmap_start(m_packet_mmap));
    DOCA_TRY(doca_eth_rxq_set_pkt_buf(m_rxq_info_cpu, m_packet_mmap, 0, cyclic_buffer_size));

    m_doca_ctx = doca_eth_rxq_as_doca_ctx(m_rxq_info_cpu);

    if (m_doca_ctx == nullptr)
        MORPHEUS_FAIL("unable to rxq as doca ctx ?");

    DOCA_TRY(doca_ctx_set_datapath_on_gpu(m_doca_ctx, context->gpu()));
    DOCA_TRY(doca_ctx_start(m_doca_ctx));
    DOCA_TRY(doca_eth_rxq_get_gpu_handle(m_rxq_info_cpu, &m_rxq_info_gpu));
}

DocaRxQueue::~DocaRxQueue()
{
    doca_error_t doca_ret;

    if (m_rxq_info_cpu != nullptr)
    {
        doca_ret = doca_ctx_stop(m_doca_ctx);
        if (doca_ret != DOCA_SUCCESS)
            LOG(WARNING) << "doca_eth_rxq_destroy failed (" << doca_ret << ")" << std::endl;

        doca_ret = doca_eth_rxq_destroy(m_rxq_info_cpu);
        if (doca_ret != DOCA_SUCCESS)
            LOG(WARNING) << "doca_eth_rxq_destroy failed (" << doca_ret << ")" << std::endl;
    }

    if (m_packet_mmap != nullptr)
    {
        doca_ret = doca_mmap_destroy(m_packet_mmap);
        if (doca_ret != DOCA_SUCCESS)
            LOG(WARNING) << "doca_mmap_destroy failed (" << doca_ret << ")" << std::endl;
    }
}

doca_eth_rxq* DocaRxQueue::rxq_info_cpu()
{
    return m_rxq_info_cpu;
}

doca_gpu_eth_rxq* DocaRxQueue::rxq_info_gpu()
{
    return m_rxq_info_gpu;
}

}  // namespace morpheus::doca
