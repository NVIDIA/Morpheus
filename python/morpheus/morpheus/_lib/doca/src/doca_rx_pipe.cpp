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

#include "morpheus/doca/doca_rx_pipe.hpp"

#include "morpheus/doca/error.hpp"

#include <doca_eth_rxq.h>
#include <doca_flow_net.h>
#include <netinet/in.h>

#include <array>
#include <cstdint>
#include <utility>

namespace morpheus::doca {

/* Create more Queues/Different Flows */
DocaRxPipe::DocaRxPipe(std::shared_ptr<DocaContext> context,
                       std::vector<std::shared_ptr<morpheus::doca::DocaRxQueue>> rxq,
                       enum doca_traffic_type const type) :
  m_context(context),
  m_rxq(std::move(rxq)),
  m_traffic_type(type),
  m_pipe(nullptr)
{
    auto rss_queues = std::array<uint16_t, MAX_QUEUE>();
    for (int idx = 0; idx < m_rxq.size(); idx++)
        doca_eth_rxq_get_flow_queue_id(m_rxq[idx]->rxq_info_cpu(), &(rss_queues[idx]));

    doca_flow_match match{};
    match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
    if (m_traffic_type == DOCA_TRAFFIC_TYPE_TCP)
    {
        match.outer.ip4.next_proto = IPPROTO_TCP;
        match.outer.l4_type_ext    = DOCA_FLOW_L4_TYPE_EXT_TCP;
    }
    else
    {
        match.outer.ip4.next_proto = IPPROTO_UDP;
        match.outer.l4_type_ext    = DOCA_FLOW_L4_TYPE_EXT_UDP;
    }

    doca_flow_fwd fwd{};
    fwd.type = DOCA_FLOW_FWD_RSS;

    if (m_traffic_type == DOCA_TRAFFIC_TYPE_TCP)
        fwd.rss_outer_flags = DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_TCP;
    else
        fwd.rss_outer_flags = DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_UDP;
    fwd.rss_queues    = rss_queues.begin();
    fwd.num_of_queues = m_rxq.size();

    doca_flow_fwd miss_fwd{};
    miss_fwd.type = DOCA_FLOW_FWD_DROP;

    doca_flow_monitor monitor{};
    monitor.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;

    struct doca_flow_pipe_cfg* pipe_cfg;
    DOCA_TRY(doca_flow_pipe_cfg_create(&pipe_cfg, context->flow_port()));
    DOCA_TRY(doca_flow_pipe_cfg_set_name(pipe_cfg, "GPU_RXQ_PIPE"));
    DOCA_TRY(doca_flow_pipe_cfg_set_enable_strict_matching(pipe_cfg, true));
    DOCA_TRY(doca_flow_pipe_cfg_set_type(pipe_cfg, DOCA_FLOW_PIPE_BASIC));
    DOCA_TRY(doca_flow_pipe_cfg_set_is_root(pipe_cfg, false));
    DOCA_TRY(doca_flow_pipe_cfg_set_match(pipe_cfg, &match, nullptr));
    DOCA_TRY(doca_flow_pipe_cfg_set_monitor(pipe_cfg, &monitor));
    DOCA_TRY(doca_flow_pipe_create(pipe_cfg, &fwd, &miss_fwd, &m_pipe));

    struct doca_flow_pipe_entry* placeholder_entry;
    DOCA_TRY(doca_flow_pipe_add_entry(
        0, m_pipe, &match, nullptr, nullptr, nullptr, DOCA_FLOW_NO_WAIT, nullptr, &placeholder_entry));
    DOCA_TRY(doca_flow_entries_process(context->flow_port(), 0, 0, 0));

    uint32_t priority_high = 1;
    uint32_t priority_low  = 3;

    doca_flow_match root_match_mask = {0};
    doca_flow_monitor root_monitor  = {};
    root_monitor.counter_type       = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;

    struct doca_flow_pipe_cfg* root_pipe_cfg;
    DOCA_TRY(doca_flow_pipe_cfg_create(&root_pipe_cfg, context->flow_port()));
    DOCA_TRY(doca_flow_pipe_cfg_set_name(root_pipe_cfg, "ROOT_PIPE"));
    DOCA_TRY(doca_flow_pipe_cfg_set_enable_strict_matching(root_pipe_cfg, true));
    DOCA_TRY(doca_flow_pipe_cfg_set_type(root_pipe_cfg, DOCA_FLOW_PIPE_CONTROL));
    DOCA_TRY(doca_flow_pipe_cfg_set_is_root(root_pipe_cfg, true));
    DOCA_TRY(doca_flow_pipe_cfg_set_match(root_pipe_cfg, nullptr, &root_match_mask));
    DOCA_TRY(doca_flow_pipe_cfg_set_monitor(root_pipe_cfg, &root_monitor));
    DOCA_TRY(doca_flow_pipe_create(root_pipe_cfg, nullptr, nullptr, &m_root_pipe));

    struct doca_flow_match root_match_gpu = {};
    struct doca_flow_fwd root_fwd_gpu     = {};
    doca_flow_pipe_entry* root_tcp_entry_gpu;

    if (m_traffic_type == DOCA_TRAFFIC_TYPE_TCP)
    {
        root_match_gpu.outer.l3_type     = DOCA_FLOW_L3_TYPE_IP4;
        root_match_gpu.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;
        root_fwd_gpu.type                = DOCA_FLOW_FWD_PIPE;
        root_fwd_gpu.next_pipe           = m_pipe;
    }
    else
    {
        root_match_gpu.outer.l3_type     = DOCA_FLOW_L3_TYPE_IP4;
        root_match_gpu.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;
        root_fwd_gpu.type                = DOCA_FLOW_FWD_PIPE;
        root_fwd_gpu.next_pipe           = m_pipe;
    }

    DOCA_TRY(doca_flow_pipe_control_add_entry(0,
                                              0, /*priority_low,*/
                                              m_root_pipe,
                                              &root_match_gpu,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              &root_fwd_gpu,
                                              nullptr,
                                              &root_tcp_entry_gpu));

    DOCA_TRY(doca_flow_entries_process(context->flow_port(), 0, 0, 0));
}

DocaRxPipe::~DocaRxPipe()
{
    doca_flow_pipe_destroy(m_root_pipe);
    doca_flow_pipe_destroy(m_pipe);
}

}  // namespace morpheus::doca
