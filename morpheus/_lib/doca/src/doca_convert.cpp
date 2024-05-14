/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "morpheus/doca/doca_kernels.hpp"
#include "morpheus/doca/doca_stages.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/raw_packet.hpp"

#include <boost/fiber/context.hpp>
#include <cuda_runtime.h>
#include <cudf/column/column.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <generic/rte_byteorder.h>
#include <glog/logging.h>
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rxcpp/rx.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#define BE_IPV4_ADDR(a, b, c, d) (RTE_BE32((a << 24) + (b << 16) + (c << 8) + d)) /* Big endian conversion */
#define ENABLE_TIMERS 0

std::optional<uint32_t> ip_to_int(std::string const& ip_address)
{
    if (ip_address.empty())
    {
        return 0;
    }

    uint8_t a, b, c, d;
    uint32_t ret;

    ret = sscanf(ip_address.c_str(), "%hhu.%hhu.%hhu.%hhu", &a, &b, &c, &d);

    printf("%u: %u %u %u %u\n", ret, a, b, c, d);

    if (ret == 4)
    {
        return BE_IPV4_ADDR(a, b, c, d);
    }

    return std::nullopt;
}

static uint64_t now_ns()
{
    struct timespec t;
    if (clock_gettime(CLOCK_REALTIME, &t) != 0)
        return 0;
    return (uint64_t)t.tv_nsec + (uint64_t)t.tv_sec * 1000 * 1000 * 1000;
}

#define debug_get_timestamp(ts) clock_gettime(CLOCK_REALTIME, (ts))

namespace morpheus {

DocaConvertStage::DocaConvertStage() :
  base_t(rxcpp::operators::map([this](sink_type_t x) {
      return this->on_data(std::move(x));
  }))
{
    cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking);
    m_stream_cpp            = rmm::cuda_stream_view(reinterpret_cast<cudaStream_t>(m_stream));
    fixed_pld_size_list_cpu = (uint32_t*)calloc(MAX_PKT_RECEIVE, sizeof(uint32_t));
    cudaMalloc((void**)&fixed_pld_size_list, MAX_PKT_RECEIVE * sizeof(uint32_t));
    for (int idx = 0; idx < MAX_PKT_RECEIVE; idx++)
        fixed_pld_size_list_cpu[idx] = MAX_PKT_SIZE;
    cudaMemcpy(fixed_pld_size_list, fixed_pld_size_list_cpu, MAX_PKT_RECEIVE * sizeof(uint32_t), cudaMemcpyDefault);

    fixed_hdr_size_list_cpu = (uint32_t*)calloc(MAX_PKT_RECEIVE, sizeof(uint32_t));
    cudaMalloc((void**)&fixed_hdr_size_list, MAX_PKT_RECEIVE * sizeof(uint32_t));
    for (int idx = 0; idx < MAX_PKT_RECEIVE; idx++)
        fixed_hdr_size_list_cpu[idx] = IP_ADDR_STRING_LEN;
    cudaMemcpy(fixed_hdr_size_list, fixed_hdr_size_list_cpu, MAX_PKT_RECEIVE * sizeof(uint32_t), cudaMemcpyDefault);
}

DocaConvertStage::~DocaConvertStage()
{
    free(fixed_pld_size_list_cpu);
    cudaFree(fixed_pld_size_list);
    free(fixed_hdr_size_list_cpu);
    cudaFree(fixed_hdr_size_list);
    cudaStreamDestroy(m_stream);
}

DocaConvertStage::source_type_t DocaConvertStage::on_data(sink_type_t x)
{
    if constexpr (std::is_same_v<sink_type_t, std::shared_ptr<RawPacketMessage>>)
    {
        return this->on_raw_packet_message(x);
    }
    // sink_type_t not supported
    else
    {
        std::string error_msg{"DocaConvertStage receives unsupported input type: " + std::string(typeid(x).name())};
        LOG(ERROR) << error_msg;
        throw std::runtime_error(error_msg);
    }
}

DocaConvertStage::source_type_t DocaConvertStage::on_raw_packet_message(sink_type_t raw_msg)
{
    auto packet_count      = raw_msg->count();
    auto max_size          = raw_msg->get_max_size();
    auto pkt_addr_list     = raw_msg->get_pkt_addr_list();
    auto pkt_hdr_size_list = raw_msg->get_pkt_hdr_size_list();
    auto pkt_pld_size_list = raw_msg->get_pkt_pld_size_list();
    auto queue_idx         = raw_msg->get_queue_idx();

    // LOG(WARNING) << "New RawPacketMessage with " << packet_count << " packets from queue id " << queue_idx;

#if ENABLE_TIMERS == 1
    const auto t0 = now_ns();
#endif
    // gather header data
    auto header_src_ip_col = doca::gather_header(
        packet_count, pkt_addr_list, pkt_hdr_size_list, pkt_pld_size_list, fixed_hdr_size_list, m_stream_cpp);

    // auto header_src_ip_col  = cudf::make_column_from_scalar(cudf::string_scalar("111.111.111.111"), packet_count);
    // auto header_src_ip_addr = header_src_ip_col->mutable_view().data<uint8_t>();
    // doca::gather_header_scalar(
    //     packet_count, pkt_addr_list, pkt_hdr_size_list, pkt_pld_size_list, header_src_ip_addr, m_stream_cpp);

#if ENABLE_TIMERS == 1
    const auto t1 = now_ns();
#endif
    // gather payload data
    auto payload_col = doca::gather_payload(
        packet_count, pkt_addr_list, pkt_hdr_size_list, pkt_pld_size_list, fixed_pld_size_list, m_stream_cpp);

#if ENABLE_TIMERS == 1
    const auto t2 = now_ns();
#endif
    std::vector<std::unique_ptr<cudf::column>> gathered_columns;
    gathered_columns.emplace_back(std::move(header_src_ip_col));
    gathered_columns.emplace_back(std::move(payload_col));

    // After this point buffers can be reused -> copies actual packets' data
    auto gathered_table = std::make_unique<cudf::table>(std::move(gathered_columns));

#if ENABLE_TIMERS == 1
    const auto t3 = now_ns();
#endif
    auto gathered_metadata = cudf::io::table_metadata();
    gathered_metadata.schema_info.emplace_back("src_ip");
    gathered_metadata.schema_info.emplace_back("data");

    auto gathered_table_w_metadata =
        cudf::io::table_with_metadata{std::move(gathered_table), std::move(gathered_metadata)};

#if ENABLE_TIMERS == 1
    const auto t4 = now_ns();
#endif
    auto meta = MessageMeta::create_from_cpp(std::move(gathered_table_w_metadata), 0);

#if ENABLE_TIMERS == 1
    const auto t5 = now_ns();
#endif
    cudaStreamSynchronize(m_stream_cpp);
#if ENABLE_TIMERS == 1
    const auto t6 = now_ns();

    LOG(WARNING) << "Queue " << queue_idx << " packets " << packet_count << " header column " << t1 - t0
                 << " payload column " << t2 - t1 << " gather columns " << t3 - t2 << " gather metadata " << t4 - t3
                 << " create_from_cpp " << t5 - t4 << " stream sync " << t6 - t5 << std::endl;
#endif

    return std::move(meta);
}

std::shared_ptr<mrc::segment::Object<DocaConvertStage>> DocaConvertStageInterfaceProxy::init(
    mrc::segment::Builder& builder, std::string const& name)
{
    return builder.construct_object<DocaConvertStage>(name);
}

}  // namespace morpheus
