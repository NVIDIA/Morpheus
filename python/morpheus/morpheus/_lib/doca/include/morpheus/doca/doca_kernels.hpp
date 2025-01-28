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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <doca_eth_rxq.h>
#include <doca_flow.h>
#include <doca_gpunetio.h>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <cstdio>
#include <memory>

namespace morpheus::doca {

uint32_t gather_sizes(int32_t packet_count, uint32_t* size_list, rmm::cuda_stream_view stream);

void gather_payload(int32_t packet_count,
                    uintptr_t* packets_buffer,
                    uint32_t* header_sizes,
                    uint32_t* payload_sizes,
                    uint8_t* dst_buff,
                    rmm::cuda_stream_view stream,
                    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

void gather_src_ip(int32_t packet_count,
                   uintptr_t* packets_buffer,
                   uint32_t* dst_buff,
                   rmm::cuda_stream_view stream,
                   rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

rmm::device_buffer sizes_to_offsets(int32_t packet_count, uint32_t* sizes_buff, rmm::cuda_stream_view stream);

int packet_receive_kernel(doca_gpu_eth_rxq* rxq_0,
                          doca_gpu_eth_rxq* rxq_1,
                          doca_gpu_semaphore_gpu* sem_0,
                          doca_gpu_semaphore_gpu* sem_1,
                          uint16_t sem_idx_0,
                          uint16_t sem_idx_1,
                          bool is_tcp,
                          uint32_t* exit_condition,
                          cudaStream_t stream);
}  // namespace morpheus::doca
