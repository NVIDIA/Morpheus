/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <rmm/cuda_stream_view.hpp>                  // for cuda_stream_view
#include <rmm/device_buffer.hpp>                     // for device_buffer
#include <rmm/mr/device/device_memory_resource.hpp>  // for device_memory_resource
#include <rmm/mr/device/per_device_resource.hpp>     // for get_current_device_resource

#include <cstddef>  // for size_t
#include <memory>

namespace morpheus::doca {

struct PacketDataBuffer
{
    PacketDataBuffer();
    PacketDataBuffer(std::size_t num_packets,
                     std::size_t header_size,
                     std::size_t payload_size,
                     std::size_t payload_sizes_size,
                     rmm::cuda_stream_view stream,
                     rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

    PacketDataBuffer(PacketDataBuffer&& other)            = default;
    PacketDataBuffer& operator=(PacketDataBuffer&& other) = default;

    std::size_t m_num_packets;
    rmm::cuda_stream_view m_stream;

    // The header buffer needs to be a shared pointer to construct a DevMemInfo
    std::shared_ptr<rmm::device_buffer> m_header_buffer;
    std::unique_ptr<rmm::device_buffer> m_payload_buffer;
    std::unique_ptr<rmm::device_buffer> m_payload_sizes_buffer;
};
}  // namespace morpheus::doca
