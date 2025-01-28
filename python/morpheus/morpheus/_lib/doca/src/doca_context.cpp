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

#include "morpheus/doca/doca_context.hpp"

#include "morpheus/doca/common.hpp"
#include "morpheus/doca/error.hpp"
#include "morpheus/utilities/error.hpp"

#include <doca_dpdk.h>
#include <doca_error.h>
#include <doca_flow.h>
#include <doca_gpunetio.h>
#include <glog/logging.h>
#include <rte_ethdev.h>
#include <rte_flow.h>
#include <rte_mbuf.h>
#include <rte_mempool.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

#define FLOW_NB_COUNTERS 524228 /* 1024 x 512 */
#define MAX_PORT_STR_LEN 128    /* Maximal length of port name */

using jobs_check_t = doca_error_t (*)(doca_devinfo*);

static doca_error_t open_doca_device_with_pci(const char* pcie_value, struct doca_dev** retval)
{
    struct doca_devinfo** dev_list;
    uint32_t nb_devs;
    doca_error_t res;
    size_t i;
    uint8_t is_addr_equal = 0;

    /* Set default return value */
    *retval = NULL;

    res = doca_devinfo_create_list(&dev_list, &nb_devs);
    if (res != DOCA_SUCCESS)
    {
        LOG(ERROR) << "Failed to load doca devices list";
        return res;
    }

    /* Search */
    for (i = 0; i < nb_devs; i++)
    {
        res = doca_devinfo_is_equal_pci_addr(dev_list[i], pcie_value, &is_addr_equal);
        if (res == DOCA_SUCCESS && is_addr_equal)
        {
            /* if device can be opened */
            res = doca_dev_open(dev_list[i], retval);
            if (res == DOCA_SUCCESS)
            {
                doca_devinfo_destroy_list(dev_list);
                return res;
            }
        }
    }

    MORPHEUS_FAIL("Matching device not found");
    res = DOCA_ERROR_NOT_FOUND;

    doca_devinfo_destroy_list(dev_list);
    return res;
}

doca_flow_port* init_doca_flow(uint16_t port_id, uint8_t rxq_num)
{
    doca_flow_port* df_port;
    rte_eth_dev_info dev_info = {nullptr};
    rte_eth_conf eth_conf     = {
            .rxmode =
            {
                    .mtu = 1024, /* Not really used, just to initialize DPDK */
            },
            .txmode =
            {
                    .offloads = RTE_ETH_TX_OFFLOAD_IPV4_CKSUM | RTE_ETH_TX_OFFLOAD_UDP_CKSUM | RTE_ETH_TX_OFFLOAD_TCP_CKSUM,
            },
    };
    rte_mempool* mp = nullptr;
    rte_eth_txconf tx_conf;
    rte_flow_error error;

    /*
     * DPDK should be initialized and started before DOCA Flow.
     * DPDK doesn't start the device without, at least, one DPDK Rx queue.
     * DOCA Flow needs to specify in advance how many Rx queues will be used by the app.
     *
     * Following lines of code can be considered the minimum WAR for this issue.
     */

    RTE_TRY(rte_eth_dev_info_get(port_id, &dev_info));
    RTE_TRY(rte_eth_dev_configure(port_id, rxq_num, rxq_num, &eth_conf));

    mp = rte_pktmbuf_pool_create("TEST", 8192, 0, 0, morpheus::doca::MAX_PKT_SIZE, rte_eth_dev_socket_id(port_id));

    if (mp == nullptr)
    {
        MORPHEUS_FAIL("rte_pktmbuf_pool_create failed.");
    }

    tx_conf = dev_info.default_txconf;
    tx_conf.offloads |= RTE_ETH_TX_OFFLOAD_IPV4_CKSUM | RTE_ETH_TX_OFFLOAD_UDP_CKSUM | RTE_ETH_TX_OFFLOAD_TCP_CKSUM;

    for (int idx = 0; idx < rxq_num; idx++)
    {
        RTE_TRY(rte_eth_rx_queue_setup(port_id, idx, 2048, rte_eth_dev_socket_id(port_id), nullptr, mp));
        RTE_TRY(rte_eth_tx_queue_setup(port_id, idx, 2048, rte_eth_dev_socket_id(port_id), &tx_conf));
    }

    RTE_TRY(rte_flow_isolate(port_id, 1, &error));
    RTE_TRY(rte_eth_dev_start(port_id));

    struct doca_flow_cfg* rxq_flow_cfg;
    DOCA_TRY(doca_flow_cfg_create(&rxq_flow_cfg));
    DOCA_TRY(doca_flow_cfg_set_pipe_queues(rxq_flow_cfg, rxq_num));
    DOCA_TRY(doca_flow_cfg_set_mode_args(rxq_flow_cfg, "vnf,hws,isolated"));
    DOCA_TRY(doca_flow_cfg_set_nr_counters(rxq_flow_cfg, FLOW_NB_COUNTERS));
    DOCA_TRY(doca_flow_init(rxq_flow_cfg));
    doca_flow_cfg_destroy(rxq_flow_cfg);

    struct doca_flow_port_cfg* port_cfg;
    char port_id_str[MAX_PORT_STR_LEN];
    DOCA_TRY(doca_flow_port_cfg_create(&port_cfg));
    snprintf(port_id_str, MAX_PORT_STR_LEN, "%d", port_id);
    DOCA_TRY(doca_flow_port_cfg_set_devargs(port_cfg, port_id_str));
    DOCA_TRY(doca_flow_port_start(port_cfg, &df_port));
    doca_flow_port_cfg_destroy(port_cfg);

    return df_port;
}

}  // namespace

namespace morpheus::doca {

DocaContext::DocaContext(std::string nic_addr, std::string gpu_addr) : m_max_queue_count(1)
{
    char* nic_addr_c = new char[nic_addr.size() + 1];
    char* gpu_addr_c = new char[gpu_addr.size() + 1];

    nic_addr_c[nic_addr.size()] = '\0';
    gpu_addr_c[gpu_addr.size()] = '\0';

    std::copy(nic_addr.begin(), nic_addr.end(), nic_addr_c);
    std::copy(gpu_addr.begin(), gpu_addr.end(), gpu_addr_c);

    m_rte_context = std::make_unique<RTEContext>();

    /* Register a logger backend for internal SDK errors and warnings */
    // DOCA_TRY(doca_log_backend_create_with_file_sdk(stderr, &m_sdk_log));
    // DOCA_TRY(doca_log_backend_set_sdk_level(m_sdk_log, DOCA_LOG_LEVEL_DEBUG));

    DOCA_TRY(open_doca_device_with_pci(nic_addr_c, &m_dev));
    DOCA_TRY(doca_dpdk_port_probe(m_dev, "dv_flow_en=2"));
    DOCA_TRY(doca_dpdk_get_first_port_id(m_dev, &m_nic_port));
    DOCA_TRY(doca_gpu_create(gpu_addr_c, &m_gpu));

    m_flow_port = init_doca_flow(m_nic_port, m_max_queue_count);

    if (m_flow_port == nullptr)
    {
        throw std::runtime_error("DOCA Flow initialization failed");
    }

    delete[] nic_addr_c;
    delete[] gpu_addr_c;
}

DocaContext::~DocaContext()
{
    doca_flow_port_stop(m_flow_port);
    doca_flow_destroy();
    if (m_gpu != nullptr)
    {
        auto doca_ret = doca_gpu_destroy(m_gpu);
        if (doca_ret != DOCA_SUCCESS)
            LOG(WARNING) << "DOCA cleanup failed (" << doca_ret << ")" << std::endl;
    }

    int ret = rte_eth_dev_stop(m_nic_port);
    if (ret != 0)
        LOG(ERROR) << "Couldn't stop DPDK port " << m_nic_port << "err " << ret;
}

doca_gpu* DocaContext::gpu()
{
    return m_gpu;
}

doca_dev* DocaContext::dev()
{
    return m_dev;
}

uint16_t DocaContext::nic_port()
{
    return m_nic_port;
}

doca_flow_port* DocaContext::flow_port()
{
    return m_flow_port;
}

}  // namespace morpheus::doca
