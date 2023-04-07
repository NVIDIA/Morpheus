#pragma once

#define DOCA_ALLOW_EXPERIMENTAL_API

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <doca_flow.h>
#include <doca_eth_rxq.h>
#include <doca_gpunetio.h>
#include <memory>

namespace morpheus {
namespace doca {

std::unique_ptr<cudf::column> integers_to_mac(
  cudf::column_view const& integers,
  rmm::cuda_stream_view stream = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

void packet_receive_kernel(
  doca_gpu_eth_rxq*       rxq_info,
  doca_gpu_semaphore_gpu* sem_in,
  int32_t                 sem_count,
  int32_t*                sem_idx,
  int32_t*                packet_count,
  int32_t*                packet_size_total,
  int32_t*                packet_sizes,
  uint8_t*                packet_buffer,
  cudaStream_t            stream
);

void packet_gather_kernel(
  doca_gpu_eth_rxq*       rxq_info,
  doca_gpu_semaphore_gpu* sem_in,
  int32_t                 sem_count,
  int32_t*                sem_idx,
  int32_t*                packet_sizes,
  uint8_t*                packet_buffer,
  uint32_t*               timestamp_out,
  int64_t*                src_mac_out,
  int64_t*                dst_mac_out,
  int64_t*                src_ip_out,
  int64_t*                dst_ip_out,
  uint16_t*               src_port_out,
  uint16_t*               dst_port_out,
  int32_t*                data_offsets_out,
  int32_t*                data_size_out,
  int32_t*                tcp_flags_out,
  int32_t*                ether_type_out,
  int32_t*                next_proto_id_out,
  char*                   data_out,
  int32_t                 data_out_size,
  cudaStream_t            stream
);

}
}
