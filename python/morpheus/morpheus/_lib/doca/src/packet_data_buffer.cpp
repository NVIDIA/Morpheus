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

#include "morpheus/doca/packet_data_buffer.hpp"

namespace morpheus::doca {

PacketDataBuffer::PacketDataBuffer() :
  m_num_packets{0},
  m_stream{rmm::cuda_stream_per_thread},
  m_header_buffer{nullptr},
  m_payload_buffer{nullptr},
  m_payload_sizes_buffer{nullptr}
{}

PacketDataBuffer::PacketDataBuffer(std::size_t num_packets,
                                   std::size_t header_size,
                                   std::size_t payload_size,
                                   std::size_t payload_sizes_size,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr) :
  m_num_packets{num_packets},
  m_stream{stream},
  m_header_buffer{std::make_shared<rmm::device_buffer>(header_size, stream, mr)},
  m_payload_buffer{std::make_unique<rmm::device_buffer>(payload_size, stream, mr)},
  m_payload_sizes_buffer{std::make_unique<rmm::device_buffer>(payload_sizes_size, stream, mr)}
{}

}  // namespace morpheus::doca
