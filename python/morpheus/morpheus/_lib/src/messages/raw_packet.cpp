/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/messages/raw_packet.hpp"

#include <pybind11/pytypes.h>

#include <memory>

// We're already including pybind11.h and don't need to include cast.
// For some reason IWYU also thinks we need array for the `isinsance` call.
// IWYU pragma: no_include <pybind11/cast.h>
// IWYU pragma: no_include <array>

namespace morpheus {

namespace py = pybind11;
using namespace py::literals;

/****** Component public implementations *******************/
/****** RawPacketMessage ****************************************/

uint32_t RawPacketMessage::count() const
{
    return m_num;
}

uint32_t RawPacketMessage::get_max_size() const
{
    return m_max_size;
}

uintptr_t RawPacketMessage::get_pkt_addr_idx(uint32_t pkt_idx) const
{
    if (pkt_idx > m_num || m_gpu_mem == true)
        return 0;
    return m_ptr_addr[pkt_idx];
}

uintptr_t RawPacketMessage::get_pkt_hdr_size_idx(uint32_t pkt_idx) const
{
    if (pkt_idx > m_num || m_gpu_mem == true)
        return 0;
    return m_ptr_hdr_size[pkt_idx];
}

uintptr_t RawPacketMessage::get_pkt_pld_size_idx(uint32_t pkt_idx) const
{
    if (pkt_idx > m_num || m_gpu_mem == true)
        return 0;
    return m_ptr_pld_size[pkt_idx];
}

uintptr_t* RawPacketMessage::get_pkt_addr_list() const
{
    return m_ptr_addr;
}

uint32_t* RawPacketMessage::get_pkt_hdr_size_list() const
{
    return m_ptr_hdr_size;
}

uint32_t* RawPacketMessage::get_pkt_pld_size_list() const
{
    return m_ptr_pld_size;
}

uint32_t RawPacketMessage::get_queue_idx() const
{
    return m_queue_idx;
}

bool RawPacketMessage::is_gpu_mem() const
{
    return m_gpu_mem;
}

std::shared_ptr<RawPacketMessage> RawPacketMessage::create_from_cpp(uint32_t num,
                                                                    uint32_t max_size,
                                                                    uintptr_t* ptr_addr,
                                                                    uint32_t* ptr_hdr_size,
                                                                    uint32_t* ptr_pld_size,
                                                                    bool gpu_mem,
                                                                    uint16_t queue_idx)
{
    return std::shared_ptr<RawPacketMessage>(
        new RawPacketMessage(num, max_size, ptr_addr, ptr_hdr_size, ptr_pld_size, gpu_mem, queue_idx));
}

RawPacketMessage::RawPacketMessage(uint32_t num_,
                                   uint32_t max_size_,
                                   uintptr_t* ptr_addr_,
                                   uint32_t* ptr_hdr_size_,
                                   uint32_t* ptr_pld_size_,
                                   bool gpu_mem_,
                                   int queue_idx_) :
  m_num(num_),
  m_max_size(max_size_),
  m_ptr_addr(ptr_addr_),
  m_ptr_hdr_size(ptr_hdr_size_),
  m_ptr_pld_size(ptr_pld_size_),
  m_gpu_mem(gpu_mem_),
  m_queue_idx(queue_idx_)
{}

}  // namespace morpheus
