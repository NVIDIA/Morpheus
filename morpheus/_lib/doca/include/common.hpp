/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <type_traits>

uint32_t const PACKETS_PER_THREAD   = 4;
uint32_t const THREADS_PER_BLOCK    = 1024; //512
uint32_t const PACKETS_PER_BLOCK    = PACKETS_PER_THREAD * THREADS_PER_BLOCK;
uint32_t const PACKET_RX_TIMEOUT_NS = 1000000; //1ms //500us

uint32_t const MAX_PKT_RECEIVE = PACKETS_PER_BLOCK;
uint32_t const MAX_PKT_SIZE    = 4096;
uint32_t const MAX_PKT_NUM     = 65536;
uint32_t const MAX_QUEUE       = 4;
uint32_t const MAX_SEM_X_QUEUE = 32;

enum doca_traffic_type {
  DOCA_TRAFFIC_TYPE_UDP = 0,
  DOCA_TRAFFIC_TYPE_TCP = 1,
};

struct packets_info {
  int32_t packet_count_out;
  int32_t payload_size_total_out;

  char *payload_buffer_out;
  int32_t *payload_sizes_out;

  int64_t *src_mac_out;
  int64_t *dst_mac_out;
  int64_t *src_ip_out;
  int64_t *dst_ip_out;
  uint16_t *src_port_out;
  uint16_t *dst_port_out;
  int32_t *tcp_flags_out;
  int32_t *ether_type_out;
  int32_t *next_proto_id_out;
  uint32_t *timestamp_out;
};