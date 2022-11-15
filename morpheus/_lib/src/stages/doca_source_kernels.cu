// #include "morpheus/doca/doca_context.hpp"

#include "morpheus/doca/common.h"
#include <doca_gpu_device.cuh>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cuda/atomic>
#include <cuda/std/chrono>
#include <memory>
#include <stdio.h>
#include <thrust/iterator/constant_iterator.h>
#include <cub/cub.cuh>

__device__ char to_hex_16(uint8_t value)
{
    return "0123456789ABCDEF"[value];
}

__device__ int64_t mac_bytes_to_int64(uint8_t* mac)
{
  return static_cast<uint64_t>(mac[0]) << 40
        | static_cast<uint64_t>(mac[1]) << 32
        | static_cast<uint32_t>(mac[2]) << 24
        | static_cast<uint32_t>(mac[3]) << 16
        | static_cast<uint32_t>(mac[4]) << 8
        | static_cast<uint32_t>(mac[5]);
}

__device__ int64_t mac_int64_to_chars(int64_t mac, char* out)
{
  uint8_t mac_0 = (mac >> 40) & (0xFF);
  out[0]  = to_hex_16(mac_0 / 16);
  out[1]  = to_hex_16(mac_0 % 16);
  out[2]  = ':';

  uint8_t mac_1 = (mac >> 32) & (0xFF);
  out[3]  = to_hex_16(mac_1 / 16);
  out[4]  = to_hex_16(mac_1 % 16);
  out[5]  = ':';

  uint8_t mac_2 = (mac >> 24) & (0xFF);
  out[6]  = to_hex_16(mac_2 / 16);
  out[7]  = to_hex_16(mac_2 % 16);
  out[8]  = ':';

  uint8_t mac_3 = (mac >> 16) & (0xFF);
  out[9]  = to_hex_16(mac_3 / 16);
  out[10] = to_hex_16(mac_3 % 16);
  out[11] = ':';

  uint8_t mac_4 = (mac >> 8) & (0xFF);
  out[12] = to_hex_16(mac_4 / 16);
  out[13] = to_hex_16(mac_4 % 16);
  out[14] = ':';

  uint8_t mac_5 = (mac >> 0) & (0xFF);
  out[15] = to_hex_16(mac_5 / 16);
  out[16] = to_hex_16(mac_5 % 16);
}

uint32_t const PACKETS_PER_THREAD = 4;
uint32_t const THREADS_PER_BLOCK = 512;
uint32_t const PACKETS_PER_BLOCK = PACKETS_PER_THREAD * THREADS_PER_BLOCK;
// uint32_t const PACKET_RX_TIMEOUT_NS = 5000000; // 5ms
uint32_t const PACKET_RX_TIMEOUT_NS = 50000000; // 50ms

__global__ void _packet_receive_kernel(
  doca_gpu_rxq_info*                              rxq_info,
  doca_gpu_semaphore_in*                          sem_in,
  uint32_t                                        sem_count,
  uint32_t*                                       sem_idx,
  uint32_t*                                       packet_count_out,
  uint32_t*                                       packet_data_size_out
)
{
  if (threadIdx.x == 0)
  {
    *packet_count_out = 0;
    *packet_data_size_out = 0;
  }
  
  __shared__ uint32_t packet_count;
  __shared__ doca_gpu_semaphore_status sem_status;
  
  uintptr_t packet_address;

  if (threadIdx.x == 0)
  {
    while (true)
    {
      auto ret = doca_gpu_device_semaphore_get_value(
        sem_in + *sem_idx,
        &sem_status,
        nullptr,
        nullptr
      );

      if (sem_status == DOCA_GPU_SEM_STATUS_FREE)
      {
        break;
      }
    }
  }

  __syncthreads();

  DOCA_GPU_VOLATILE(packet_count) = 0;

  __syncthreads();

  auto ret = doca_gpu_device_receive_block(
    rxq_info,
    PACKETS_PER_BLOCK,
    PACKET_RX_TIMEOUT_NS,
    nullptr,
    false,
    &packet_count,
    &packet_address
  );

  __threadfence();
  __syncthreads();

  if (packet_count == 0) {
    return;
  }

  __shared__ uint32_t stride_start_idx;

  if (threadIdx.x == 0) {
    *packet_count_out = packet_count;
    stride_start_idx = doca_gpu_device_comm_buf_get_stride_idx(
      &(rxq_info->comm_buf),
      packet_address
    );
  }

  __syncthreads();

  for (auto i = 0; i < PACKETS_PER_THREAD; i++)
  {
    auto packet_idx = threadIdx.x * PACKETS_PER_THREAD + i;

    if (packet_idx >= packet_count) {
      continue;
    }

    uint8_t *packet = doca_gpu_device_comm_buf_get_stride_addr(
      &(rxq_info->comm_buf),
      stride_start_idx + packet_idx
    );

    rte_ether_hdr* packet_l2;
    rte_ipv4_hdr*  packet_l3;
    rte_tcp_hdr*   packet_l4;
    uint8_t*       packet_data;

    get_packet_tcp_headers(
      packet,
      &packet_l2,
      &packet_l3,
      &packet_l4,
      &packet_data
    );

    auto total_length = static_cast<int32_t>(BYTE_SWAP16(packet_l3->total_length));
    auto data_size = total_length - static_cast<int32_t>(packet_l4->dt_off * sizeof(int32_t));

    atomicAdd(packet_data_size_out, data_size);

    // printf("packet_idx(%d) data_size(%d) atom\n", packet_idx, data_size);
  }

  __syncthreads();

  if (threadIdx.x == 0)
  {
    doca_gpu_device_semaphore_update(
      sem_in + *sem_idx,
      DOCA_GPU_SEM_STATUS_HOLD,
      packet_count,
      packet_address
    );
    // doca_gpu_device_semaphore_update_status(
    //   sem_in + *sem_idx,
    //   DOCA_GPU_SEM_STATUS_HOLD
    // );
  }

  __threadfence();
  __syncthreads();
}

__global__ void _packet_gather_kernel(
  doca_gpu_rxq_info*                              rxq_info,
  doca_gpu_semaphore_in*                          sem_in,
  uint32_t                                        sem_count,
  uint32_t*                                       sem_idx,
  uint64_t*                                       timestamp_out,
  int64_t*                                        src_mac_out,
  int64_t*                                        dst_mac_out,
  int64_t*                                        src_ip_out,
  int64_t*                                        dst_ip_out,
  uint16_t*                                       src_port_out,
  uint16_t*                                       dst_port_out,
  int32_t*                                        data_offsets_out,
  char*                                           data_out
)
{
  // Specialize BlockScan for a 1D block of 128 threads of type int
  using BlockScan = cub::BlockScan<int32_t, THREADS_PER_BLOCK>;

  // Allocate shared memory for BlockScan
  __shared__ typename BlockScan::TempStorage temp_storage;

  __shared__ doca_gpu_semaphore_status sem_status;
	__shared__ uint32_t packet_count;
  __shared__ uintptr_t packet_address;

  if (threadIdx.x == 0) {

    doca_error_t ret;
    do
    {
      ret = doca_gpu_device_semaphore_get_value_status(
        sem_in + *sem_idx,
        DOCA_GPU_SEM_STATUS_HOLD,
        &sem_status,
        &packet_count,
        &packet_address);

      // auto ret = doca_gpu_device_semaphore_get_value(
      //   sem_in + *sem_idx,
      //   &sem_status,
      //   &packet_count,
      //   &packet_address
      // );

      // if (sem_status == DOCA_GPU_SEM_STATUS_HOLD)
      // {
      //   break;
      // }
    } while(ret == DOCA_ERROR_NOT_FOUND and sem_status != DOCA_GPU_SEM_STATUS_HOLD);
  }

  // __syncthreads();

  // auto ret = doca_gpu_device_semaphore_get_value(
  //   sem_in + *sem_idx,
  //   &sem_status,
  //   &packet_count,
  //   &packet_address
  // );

  __syncthreads();

  __shared__ uint32_t stride_start_idx;

  if (threadIdx.x == 0) {
    stride_start_idx = doca_gpu_device_comm_buf_get_stride_idx(
      &(rxq_info->comm_buf),
      packet_address
    );
  }

  __syncthreads();

  int32_t data_offsets[PACKETS_PER_THREAD];

  for (auto i = 0; i < PACKETS_PER_THREAD; i++)
  {
    auto packet_idx = threadIdx.x * PACKETS_PER_THREAD + i;

    if (packet_idx >= packet_count) {
      continue;
      data_offsets[i] = 0;
    }

    uint8_t *packet = doca_gpu_device_comm_buf_get_stride_addr(
      &(rxq_info->comm_buf),
      stride_start_idx + packet_idx
    );

    rte_ether_hdr* packet_l2;
    rte_ipv4_hdr*  packet_l3;
    rte_tcp_hdr*   packet_l4;
    uint8_t*       packet_data;

    get_packet_tcp_headers(
      packet,
      &packet_l2,
      &packet_l3,
      &packet_l4,
      &packet_data
    );

    auto total_length = static_cast<int32_t>(BYTE_SWAP16(packet_l3->total_length));
    auto data_size = total_length - static_cast<int32_t>(packet_l4->dt_off * sizeof(int32_t));

    data_offsets[i] = data_size;

    // mac address
    auto src_mac = packet_l2->s_addr.addr_bytes; // 6 bytes
    auto dst_mac = packet_l2->d_addr.addr_bytes; // 6 bytes

    src_mac_out[packet_idx] = mac_bytes_to_int64(src_mac);
    dst_mac_out[packet_idx] = mac_bytes_to_int64(dst_mac);

    // ip address
    auto src_address  = packet_l3->src_addr;
    auto dst_address  = packet_l3->dst_addr;

    auto src_address_rev = (src_address & 0x000000ff) << 24
                          | (src_address & 0x0000ff00) << 8
                          | (src_address & 0x00ff0000) >> 8
                          | (src_address & 0xff000000) >> 24;

    auto dst_address_rev = (dst_address & 0x000000ff) << 24
                          | (dst_address & 0x0000ff00) << 8
                          | (dst_address & 0x00ff0000) >> 8
                          | (dst_address & 0xff000000) >> 24;

    src_ip_out[packet_idx] = src_address_rev;
    dst_ip_out[packet_idx] = dst_address_rev;

    // ports
    auto src_port     = BYTE_SWAP16(packet_l4->src_port);
    auto dst_port     = BYTE_SWAP16(packet_l4->dst_port);

    src_port_out[packet_idx] = src_port;
    dst_port_out[packet_idx] = dst_port;
  }

  BlockScan(temp_storage).ExclusiveSum(data_offsets, data_offsets);

  __syncthreads();

  for (auto i = 0; i < PACKETS_PER_THREAD; i++)
  {
    auto packet_idx = threadIdx.x * PACKETS_PER_THREAD + i;

    if (packet_idx >= packet_count) {
      continue;
    }

    uint8_t *packet = doca_gpu_device_comm_buf_get_stride_addr(
      &(rxq_info->comm_buf),
      stride_start_idx + packet_idx
    );

    rte_ether_hdr* packet_l2;
    rte_ipv4_hdr*  packet_l3;
    rte_tcp_hdr*   packet_l4;
    uint8_t*       packet_data;

    get_packet_tcp_headers(
      packet,
      &packet_l2,
      &packet_l3,
      &packet_l4,
      &packet_data
    );

    auto total_length = static_cast<int32_t>(BYTE_SWAP16(packet_l3->total_length));
    auto data_size = total_length - static_cast<int32_t>(packet_l4->dt_off * sizeof(int32_t));

    // printf("packet_idx(%d) data_offset(%d)\n", packet_idx, data_offsets[i]);

    data_offsets_out[packet_idx] = data_offsets[i];

    for (auto data_idx = 0; data_idx < data_size; data_idx++)
    {
      data_out[data_offsets[i] + data_idx] = packet_data[data_idx];
    }
  }

  __syncthreads();

  if (threadIdx.x == 0)
  {
    doca_gpu_device_semaphore_update_status(
      sem_in + *sem_idx,
      DOCA_GPU_SEM_STATUS_FREE
    );
  }

  // // if (threadIdx.x == 0) {
  // //   printf("kernel gather: started\n");
  // // }

  // __shared__ doca_gpu_semaphore_status sem_status;
	// __shared__ uint32_t packet_count;
  // __shared__ uint32_t payload_offset_total;

	// uintptr_t packet_address;

  // uint32_t sem_idx = *sem_idx_begin;

  // // ===== WAIT FOR HELD SEM ======================================================================

  // // don't need to wait because we know which sems to process.
  // // rule 1: sem at sem_idx_begin must be processed, because we wouldn't be here if there weren't at least one sem to process.
  // // rule 2: all sems up to sem_idx_end (exclusive) must be processed.
  // // rule 3: if sem_idx_begin == sem_idx_end, sem_idx_begin still gets processed due to rule 1.

  // __shared__ uint32_t packet_offset;

  // if (threadIdx.x == 0)
  // {
  //   packet_offset = 0;
  //   payload_offset_total = 0;
  // }

  // __syncthreads();

  // while (*exit_flag == false)
  // {
  //   DOCA_GPU_VOLATILE(packet_count) = 0;

  //   __syncthreads();

  //   // get sem info
  //   auto ret = doca_gpu_device_semaphore_get_value(
  //     sem_in + sem_idx,
  //     &sem_status,
  //     &packet_count,
  //     &packet_address
  //   );

  //   if (ret != DOCA_SUCCESS)
  //   {
  //     *exit_flag = true;
  //     continue;
  //   }

  //   __syncthreads();

  //   // copy packets to dataframe

  //   __shared__ uint32_t stride_start_idx;

  //   if (threadIdx.x == 0) {
	// 		stride_start_idx = doca_gpu_device_comm_buf_get_stride_idx(
  //       &(rxq_info->comm_buf),
  //       packet_address
  //     );
  //   }

  //   __syncthreads();

  //   // Obtain a segment of consecutive items that are blocked across threads
  //   uint32_t payload_offsets[PACKETS_PER_THREAD];

  //   for (auto i = 0; i < PACKETS_PER_THREAD; i++)
  //   {
  //     auto packet_idx = threadIdx.x * PACKETS_PER_THREAD + i;

  //     if (packet_idx >= packet_count) {
  //       payload_offsets[i] = 0;
  //       continue;
  //     }

  //     uint8_t *packet = doca_gpu_device_comm_buf_get_stride_addr(
  //       &(rxq_info->comm_buf),
  //       stride_start_idx + packet_idx
  //     );

  //     rte_ether_hdr* packet_l2;
  //     rte_ipv4_hdr*  packet_l3;
  //     rte_tcp_hdr*   packet_l4;
  //     uint8_t*       packet_payload;

  //     get_packet_tcp_headers(
  //       packet,
  //       &packet_l2,
  //       &packet_l3,
  //       &packet_l4,
  //       &packet_payload
  //     );

  //     auto packet_out_idx = packet_offset + packet_idx;

  //     timestamp_out[packet_out_idx] = cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds>(cuda::std::chrono::system_clock::now().time_since_epoch()).count();

  //     auto total_length = BYTE_SWAP16(packet_l3->total_length);
  //     auto payload_size = total_length - (packet_l4->dt_off * sizeof(int));

  //     if (payload_size > 0)
  //     {
  //       printf("payload_size %d\n", payload_size);
  //     }

  //     payload_size_out[packet_out_idx] = payload_size;
  //     payload_offsets[i] = payload_size;

  //     // mac address printing works
  //     auto src_mac = packet_l2->s_addr.addr_bytes; // 6 bytes
  //     auto dst_mac = packet_l2->d_addr.addr_bytes; // 6 bytes

  //     src_mac_out[packet_out_idx] = mac_bytes_to_int64(src_mac);
  //     dst_mac_out[packet_out_idx] = mac_bytes_to_int64(dst_mac);

  //     // ip address printing works
  //     auto src_address  = packet_l3->src_addr;
  //     auto dst_address  = packet_l3->dst_addr;
  //     auto src_port     = BYTE_SWAP16(packet_l4->src_port);
  //     auto dst_port     = BYTE_SWAP16(packet_l4->dst_port);

  //     // reverse the bytes so int64->ip string kernel works properly.

  //     auto src_address_rev = (src_address & 0x000000ff) << 24
  //                          | (src_address & 0x0000ff00) << 8
  //                          | (src_address & 0x00ff0000) >> 8
  //                          | (src_address & 0xff000000) >> 24;

  //     auto dst_address_rev = (dst_address & 0x000000ff) << 24
  //                          | (dst_address & 0x0000ff00) << 8
  //                          | (dst_address & 0x00ff0000) >> 8
  //                          | (dst_address & 0xff000000) >> 24;

  //     src_ip_out[packet_out_idx] = src_address_rev;
  //     dst_ip_out[packet_out_idx] = dst_address_rev;

  //     src_port_out[packet_out_idx] = src_port;
  //     dst_port_out[packet_out_idx] = dst_port;
  //   }

  //   __syncthreads();

  //   uint32_t payload_block_offset;

  //   // Collectively compute the block-wide exclusive prefix sum
  //   BlockScan(temp_storage).ExclusiveSum(payload_offsets, payload_offsets, payload_block_offset);

  //   for (auto i = 0; i < PACKETS_PER_THREAD; i++)
  //   {
  //     auto packet_idx = threadIdx.x * PACKETS_PER_THREAD + i;

  //     if (packet_idx >= packet_count) {
  //       continue;
  //     }

  //     uint8_t *packet = doca_gpu_device_comm_buf_get_stride_addr(
  //       &(rxq_info->comm_buf),
  //       stride_start_idx + packet_idx
  //     );

  //     rte_ether_hdr* packet_l2;
  //     rte_ipv4_hdr*  packet_l3;
  //     rte_tcp_hdr*   packet_l4;
  //     uint8_t*       packet_payload;

  //     get_packet_tcp_headers(
  //       packet,
  //       &packet_l2,
  //       &packet_l3,
  //       &packet_l4,
  //       &packet_payload
  //     );

  //     auto total_length = BYTE_SWAP16(packet_l3->total_length);
  //     auto payload_size = total_length - (packet_l4->dt_off * sizeof(int));

  //     auto payload_offset = payload_offset_total + payload_offsets[i];

  //     for (auto j = 0; j < payload_size; j++)
  //     {
  //       // payload_data_out[payload_offset + j] = packet_payload[j];
  //     }
  //   }

  //   if(threadIdx.x == 0)
  //   {
  //     payload_offset_total += payload_block_offset;
  //   }

  //   // release sem

  //   if (threadIdx.x == 0)
  //   {
  //     // printf("kernel gather: setting sem %d to free\n", sem_idx);

  //     doca_gpu_device_semaphore_update_status(
  //       sem_in + sem_idx,
  //       DOCA_GPU_SEM_STATUS_FREE
  //     );

  //     packet_offset += packet_count;
  //   }

  //   __syncthreads();

  //   // determine if the next sem should be processed

  //   sem_idx = (sem_idx + 1) % sem_count;

  //   if (sem_idx == *sem_idx_end)
  //   {
  //     break;
  //   }
  // }

  // *sem_idx_begin = *sem_idx_end;

  // // if (threadIdx.x == 0) {
  // //   printf("kernel gather: done\n");
  // // }

  // __syncthreads();
}

namespace morpheus {
namespace doca {

namespace {

struct integers_to_mac_fn {
  cudf::column_device_view const d_column;
  int32_t const* d_offsets;
  char* d_chars;

  __device__ void operator()(cudf::size_type idx)
  {
    int64_t mac_address = d_column.element<int64_t>(idx);
    char* out_ptr       = d_chars + d_offsets[idx];
    
    mac_int64_to_chars(mac_address, out_ptr);
  }
};

}

std::unique_ptr<cudf::column> integers_to_mac(
  cudf::column_view const& integers,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr
)
{
  CUDF_EXPECTS(integers.type().id() == cudf::type_id::INT64, "Input column must be type_id::INT64 type");
  CUDF_EXPECTS(integers.null_count() == 0, "integers_to_mac does not support null values.");

  cudf::size_type strings_count = integers.size();

  if (strings_count == 0)
  {
    return cudf::make_empty_column(cudf::type_id::STRING);
  }

  auto offsets_transformer_itr = thrust::constant_iterator<int32_t>(17);
  auto offsets_column = cudf::strings::detail::make_offsets_child_column(
    offsets_transformer_itr,
    offsets_transformer_itr + strings_count,
    stream,
    mr
  );

  auto d_offsets = offsets_column->view().data<int32_t>();

  auto column   = cudf::column_device_view::create(integers, stream);
  auto d_column = *column;

  auto const bytes =
    cudf::detail::get_value<int32_t>(offsets_column->view(), strings_count, stream);

  auto chars_column = cudf::strings::detail::create_chars_child_column(bytes, stream, mr);
  auto d_chars      = chars_column->mutable_view().data<char>();

  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<cudf::size_type>(0),
    strings_count,
    integers_to_mac_fn{d_column, d_offsets, d_chars}
  );

  return cudf::make_strings_column(strings_count,
    std::move(offsets_column),
    std::move(chars_column),
    0,
    {});
}

struct picker {
  uint32_t* lengths;
  __device__ uint32_t operator()(cudf::size_type idx){
    if (lengths[idx] > 0)
    {
      printf("pdl: %d\n", lengths[idx]);
    }
    return lengths[idx];
  }
};

std::unique_ptr<cudf::column> packet_data_to_column(
  cudf::size_type packet_count,
  rmm::device_uvector<char> && packet_data_chars,
  rmm::device_uvector<uint32_t> && packet_data_lengths,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto offsets_transformer_itr = thrust::make_transform_iterator(
    thrust::make_counting_iterator<int32_t>(0),
    picker{packet_data_lengths.data()}
    // [data_lengths = packet_data_lengths.data()] __device__(cudf::size_type idx) {
    //   return data_lengths[idx];
    // }
  );

  auto payload_offsets_column = cudf::strings::detail::make_offsets_child_column(
    offsets_transformer_itr,
    offsets_transformer_itr + packet_count,
    stream,
    mr
  );

  stream.synchronize();

  auto packet_data_chars_size = packet_data_chars.size();
  auto packet_data_chars_col  = std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_to_id<char>()},
    packet_data_chars_size,
    packet_data_chars.release());

  uint32_t last_offset;

  cudaMemcpy(&last_offset, payload_offsets_column->view().data<uint32_t>() + packet_count - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);

  std::cout << "last offset: " << last_offset << " "
            << "chars size: " << packet_data_chars_size
            << std::endl;

  return cudf::make_strings_column(
    packet_count,
    std::move(payload_offsets_column),
    std::move(packet_data_chars_col),
    0,
    {}
  );
}

void packet_receive_kernel(
  doca_gpu_rxq_info*                              rxq_info,
  doca_gpu_semaphore_in*                          sem_in,
  uint32_t                                        sem_count,
  uint32_t*                                       sem_idx,
  uint32_t*                                       packet_count,
  uint32_t*                                       packet_data_size,
  cudaStream_t                                    stream
)
{
  _packet_receive_kernel<<<1, THREADS_PER_BLOCK, 0, stream>>>(
    rxq_info,
    sem_in,
    sem_count,
    sem_idx,
    packet_count,
    packet_data_size
  );
}

void packet_gather_kernel(
  doca_gpu_rxq_info*                              rxq_info,
  doca_gpu_semaphore_in*                          sem_in,
  uint32_t                                        sem_count,
  uint32_t*                                       sem_idx,
  uint64_t*                                       timestamp_out,
  int64_t*                                        src_mac_out,
  int64_t*                                        dst_mac_out,
  int64_t*                                        src_ip_out,
  int64_t*                                        dst_ip_out,
  uint16_t*                                       src_port_out,
  uint16_t*                                       dst_port_out,
  int32_t*                                        data_offsets_out,
  char*                                           data_out,
  cudaStream_t                                    stream
)
{
  _packet_gather_kernel<<<1, THREADS_PER_BLOCK, 0, stream>>>(
    rxq_info,
    sem_in,
    sem_count,
    sem_idx,
    timestamp_out,
    src_mac_out,
    dst_mac_out,
    src_ip_out,
    dst_ip_out,
    src_port_out,
    dst_port_out,
    data_offsets_out,
    data_out
  );
}

}
}
