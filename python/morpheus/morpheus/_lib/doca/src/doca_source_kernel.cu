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

#include "morpheus/doca/common.hpp"
#include "morpheus/doca/packets.hpp"
#include "morpheus/utilities/error.hpp"

#include <cub/cub.cuh>
#include <cuda/std/chrono>
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <doca_eth_rxq.h>
#include <doca_gpunetio.h>
#include <doca_gpunetio_dev_buf.cuh>
#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_gpunetio_dev_sem.cuh>
#include <rmm/exec_policy.hpp>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_tcp.h>
#include <stdio.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <memory>

#define DEVICE_GET_TIME(globaltimer) asm volatile("mov.u64 %0, %globaltimer;" : "=l"(globaltimer))

using namespace morpheus::doca;

__global__ void _packet_receive_kernel(doca_gpu_eth_rxq* rxq_0,
                                       doca_gpu_eth_rxq* rxq_1,
                                       doca_gpu_semaphore_gpu* sem_0,
                                       doca_gpu_semaphore_gpu* sem_1,
                                       uint16_t sem_idx_0,
                                       uint16_t sem_idx_1,
                                       const bool is_tcp,
                                       uint32_t* exit_condition)
{
    __shared__ uint32_t packet_count_received;
    __shared__ uint64_t packet_offset_received;
    __shared__ struct packets_info* pkt_info;
#if RUN_PERSISTENT
    doca_gpu_semaphore_status sem_status;
#endif
    doca_gpu_buf* buf_ptr;
    uintptr_t buf_addr;
    doca_error_t doca_ret;
    struct eth_ip_tcp_hdr* hdr_tcp;
    struct eth_ip_udp_hdr* hdr_udp;
    uint8_t* payload;
    doca_gpu_eth_rxq* rxq;
    doca_gpu_semaphore_gpu* sem;
    uint16_t sem_idx;
    uint32_t pkt_idx = threadIdx.x;
    // unsigned long long rx_start = 0, rx_stop = 0, pkt_proc = 0, reduce_stop =0, reduce_start = 0;

    if (blockIdx.x == 0)
    {
        rxq     = rxq_0;
        sem     = sem_0;
        sem_idx = sem_idx_0;
    }
    else
    {
        rxq     = rxq_1;
        sem     = sem_1;
        sem_idx = sem_idx_1;
    }

    // Initial semaphore index 0, assume it's free!
    doca_ret = doca_gpu_dev_semaphore_get_custom_info_addr(sem, sem_idx, (void**)&pkt_info);
    if (doca_ret != DOCA_SUCCESS)
    {
        printf("Error %d doca_gpu_dev_semaphore_get_custom_info_addr\n", doca_ret);
        DOCA_GPUNETIO_VOLATILE(*exit_condition) = 1;
        return;
    }

    if (threadIdx.x == 0)
    {
        DOCA_GPUNETIO_VOLATILE(pkt_info->packet_count_out) = 0;
        DOCA_GPUNETIO_VOLATILE(packet_count_received)      = 0;
    }
    __syncthreads();

    // do {
    // if (threadIdx.x == 0) DEVICE_GET_TIME(rx_start);
    doca_ret = doca_gpu_dev_eth_rxq_receive_block(
        rxq, PACKETS_PER_BLOCK, PACKET_RX_TIMEOUT_NS, &packet_count_received, &packet_offset_received);
    if (doca_ret != DOCA_SUCCESS) [[unlikely]]
    {
        DOCA_GPUNETIO_VOLATILE(*exit_condition) = 1;
        return;
    }
    __threadfence();
    if (DOCA_GPUNETIO_VOLATILE(packet_count_received) == 0)
        return;

    while (pkt_idx < DOCA_GPUNETIO_VOLATILE(packet_count_received))
    {
        doca_ret =
            doca_gpu_dev_eth_rxq_get_buf(rxq, DOCA_GPUNETIO_VOLATILE(packet_offset_received) + pkt_idx, &buf_ptr);
        if (doca_ret != DOCA_SUCCESS) [[unlikely]]
        {
            DOCA_GPUNETIO_VOLATILE(*exit_condition) = 1;
            return;
        }

        doca_ret = doca_gpu_dev_buf_get_addr(buf_ptr, &buf_addr);
        if (doca_ret != DOCA_SUCCESS) [[unlikely]]
        {
            DOCA_GPUNETIO_VOLATILE(*exit_condition) = 1;
            return;
        }

        pkt_info->pkt_addr[pkt_idx] = buf_addr;
        if (is_tcp)
        {
            raw_to_tcp(buf_addr, &hdr_tcp, &payload);
            pkt_info->pkt_hdr_size[pkt_idx] = TCP_HDR_SIZE;
            pkt_info->pkt_pld_size[pkt_idx] = get_payload_tcp_size(hdr_tcp->l3_hdr, hdr_tcp->l4_hdr);
        }
        else
        {
            raw_to_udp(buf_addr, &hdr_udp, &payload);
            pkt_info->pkt_hdr_size[pkt_idx] = UDP_HDR_SIZE;
            pkt_info->pkt_pld_size[pkt_idx] = get_payload_udp_size(hdr_udp->l3_hdr, hdr_udp->l4_hdr);
        }

        pkt_idx += blockDim.x;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        DOCA_GPUNETIO_VOLATILE(pkt_info->packet_count_out) = packet_count_received;
        DOCA_GPUNETIO_VOLATILE(packet_count_received)      = 0;
        doca_ret = doca_gpu_dev_semaphore_set_status(sem, sem_idx, DOCA_GPU_SEMAPHORE_STATUS_READY);
        if (doca_ret != DOCA_SUCCESS)
        {
            printf("Error %d doca_gpu_dev_semaphore_set_status\n", doca_ret);
            DOCA_GPUNETIO_VOLATILE(*exit_condition) = 1;
        }

        // printf("CUDA rx time %ld proc time %ld pkt conv %ld block reduce %ld\n",
        //         rx_stop - rx_start,
        //         pkt_proc - rx_stop,
        //         reduce_start - rx_stop,
        //         reduce_stop - reduce_start);
    }
    __syncthreads();

#if RUN_PERSISTENT
    // sem_idx = (sem_idx+1)%MAX_SEM_X_QUEUE;

    // Get packets' info from next semaphore
    // if (threadIdx.x == 0) {
    // do {
    //     doca_ret = doca_gpu_dev_semaphore_get_status(sem, sem_idx, &sem_status);
    //     if (doca_ret != DOCA_SUCCESS) {
    //         printf("Error %d doca_gpu_dev_semaphore_get_status\n", doca_ret);
    //         DOCA_GPUNETIO_VOLATILE(*exit_condition) = 1;
    //         break;
    //     }

    //     if (sem_status == DOCA_GPU_SEMAPHORE_STATUS_FREE) {
    //         doca_ret = doca_gpu_dev_semaphore_get_custom_info_addr(sem, sem_idx, (void **)&pkt_info);
    //         if (doca_ret != DOCA_SUCCESS) {
    //             printf("Error %d doca_gpu_dev_semaphore_get_custom_info_addr\n", doca_ret);
    //             DOCA_GPUNETIO_VOLATILE(*exit_condition) = 1;
    //         }

    //         DOCA_GPUNETIO_VOLATILE(pkt_info->packet_count_out) = 0;
    //         DOCA_GPUNETIO_VOLATILE(pkt_info->payload_size_total_out) = 0;
    //         DOCA_GPUNETIO_VOLATILE(packet_count_received) = 0;

    //         break;
    //     }
    // } while (DOCA_GPUNETIO_VOLATILE(*exit_condition) == 0);
    // }
    // __syncthreads();
    // } while (DOCA_GPUNETIO_VOLATILE(*exit_condition) == 0)

    if (threadIdx.x == 0)
        doca_gpu_dev_sem_set_status(sem_in, *sem_idx, DOCA_GPU_SEMAPHORE_STATUS_FREE);
        // __threadfence();
        // __syncthreads();
#endif
}

namespace morpheus {
namespace doca {

int packet_receive_kernel(doca_gpu_eth_rxq* rxq_0,
                          doca_gpu_eth_rxq* rxq_1,
                          doca_gpu_semaphore_gpu* sem_0,
                          doca_gpu_semaphore_gpu* sem_1,
                          uint16_t sem_idx_0,
                          uint16_t sem_idx_1,
                          bool is_tcp,
                          uint32_t* exit_condition,
                          cudaStream_t stream)
{
    cudaError_t result = cudaSuccess;

    _packet_receive_kernel<<<MAX_QUEUE, THREADS_PER_BLOCK, 0, stream>>>(
        rxq_0, rxq_1, sem_0, sem_1, sem_idx_0, sem_idx_1, is_tcp, exit_condition);

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (cudaSuccess != result)
    {
        fprintf(stderr, "[%s:%d] cuda failed with %s\n", __FILE__, __LINE__, cudaGetErrorString(result));
        return -1;
    }

    return 0;
}

}  // namespace doca
}  // namespace morpheus
