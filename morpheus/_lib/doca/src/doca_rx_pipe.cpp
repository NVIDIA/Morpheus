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

#include "doca_rx_pipe.hpp"

#include <glog/logging.h>
#include <netinet/in.h>

namespace morpheus::doca {

DocaRxPipe::DocaRxPipe(std::shared_ptr<DocaContext> context, std::shared_ptr<DocaRxQueue> rxq) :
  m_context(context),
  m_rxq(rxq),
  m_pipe(nullptr)
{
    auto rss_queues = std::array<uint16_t, 1>();
    doca_eth_rxq_get_flow_queue_id(rxq->rxq_info_cpu(), &(rss_queues[0]));

    doca_flow_match match_mask{0};
    doca_flow_match match{};
    match.outer.l3_type        = DOCA_FLOW_L3_TYPE_IP4;
    match.outer.ip4.next_proto = IPPROTO_TCP;
    match.outer.l4_type_ext    = DOCA_FLOW_L4_TYPE_EXT_TCP;

    doca_flow_fwd fwd{};
    fwd.type            = DOCA_FLOW_FWD_RSS;
    fwd.rss_outer_flags = DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_TCP;
    fwd.rss_queues      = rss_queues.begin();
    fwd.num_of_queues   = 1;

    doca_flow_fwd miss_fwd{};
    miss_fwd.type = DOCA_FLOW_FWD_DROP;

    doca_flow_monitor monitor{};
    monitor.flags = DOCA_FLOW_MONITOR_COUNT;

    doca_flow_pipe_cfg pipe_cfg{};
    pipe_cfg.attr.name       = "GPU_RXQ_TCP_PIPE";
    pipe_cfg.attr.type       = DOCA_FLOW_PIPE_BASIC;
    pipe_cfg.attr.nb_actions = 0;
    pipe_cfg.attr.is_root    = false;
    pipe_cfg.match           = &match;
    pipe_cfg.match_mask      = &match_mask;
    pipe_cfg.monitor         = &monitor;
    pipe_cfg.port            = context->flow_port();

    DOCA_TRY(doca_flow_pipe_create(&pipe_cfg, &fwd, &miss_fwd, &m_pipe));

    doca_flow_pipe_entry* placeholder_entry = nullptr;
    DOCA_TRY(doca_flow_pipe_add_entry(0, m_pipe, nullptr, nullptr, nullptr, nullptr, 0, nullptr, &placeholder_entry));
    DOCA_TRY(doca_flow_entries_process(context->flow_port(), 0, 0, 0));

    uint32_t priority_high = 1;
    uint32_t priority_low  = 3;

    doca_flow_match root_match_mask = {0};
    doca_flow_monitor root_monitor  = {};
    root_monitor.flags              = DOCA_FLOW_MONITOR_COUNT;

    doca_flow_pipe_cfg root_pipe_cfg = {};
    root_pipe_cfg.attr.name          = "ROOT_PIPE";
    root_pipe_cfg.attr.is_root       = true;
    root_pipe_cfg.attr.type          = DOCA_FLOW_PIPE_CONTROL;
    root_pipe_cfg.monitor            = &root_monitor;
    root_pipe_cfg.match_mask         = &root_match_mask;
    root_pipe_cfg.port               = context->flow_port();

    DOCA_TRY(doca_flow_pipe_create(&root_pipe_cfg, nullptr, nullptr, &m_root_pipe));

    struct doca_flow_match tcp_match_gpu = {};
    tcp_match_gpu.outer.l3_type          = DOCA_FLOW_L3_TYPE_IP4;
    tcp_match_gpu.outer.l4_type_ext      = DOCA_FLOW_L4_TYPE_EXT_TCP;

    struct doca_flow_fwd tcp_fwd_gpu = {};
    tcp_fwd_gpu.type                 = DOCA_FLOW_FWD_PIPE;
    tcp_fwd_gpu.next_pipe            = m_pipe;

    doca_flow_pipe_entry* root_tcp_entry_gpu;

    DOCA_TRY(doca_flow_pipe_control_add_entry(0,
                                              priority_low,
                                              m_root_pipe,
                                              &tcp_match_gpu,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              &tcp_fwd_gpu,
                                              &root_tcp_entry_gpu));

    DOCA_TRY(doca_flow_entries_process(context->flow_port(), 0, 0, 0));
}

DocaRxPipe::~DocaRxPipe()
{
    doca_flow_pipe_destroy(m_root_pipe);
    doca_flow_pipe_destroy(m_pipe);
}

}  // namespace morpheus::doca
