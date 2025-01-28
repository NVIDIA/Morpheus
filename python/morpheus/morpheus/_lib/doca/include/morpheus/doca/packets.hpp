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

#pragma once

#include "morpheus/doca/common.hpp"

#define ETHER_ADDR_LEN 6 /**< Length of Ethernet address. */

#define BYTE_SWAP16(v) ((((uint16_t)(v)&UINT16_C(0x00ff)) << 8) | (((uint16_t)(v)&UINT16_C(0xff00)) >> 8))

#define TCP_PROTOCOL_ID 0x6
#define UDP_PROTOCOL_ID 0x11
#define TIMEOUT_NS 500000  // 500us
#define RTE_IPV4_HDR_IHL_MASK (0x0f)
#define RTE_IPV4_IHL_MULTIPLIER (4)

// Allow naming and c arrays for compatibility with existing code
// NOLINTBEGIN(readability-identifier-naming)
// NOLINTBEGIN(modernize-avoid-c-arrays)

enum tcp_flags
{
    TCP_FLAG_FIN = (1 << 0),
    /* set tcp packet with Fin flag */
    TCP_FLAG_SYN = (1 << 1),
    /* set tcp packet with Syn flag */
    TCP_FLAG_RST = (1 << 2),
    /* set tcp packet with Rst flag */
    TCP_FLAG_PSH = (1 << 3),
    /* set tcp packet with Psh flag */
    TCP_FLAG_ACK = (1 << 4),
    /* set tcp packet with Ack flag */
    TCP_FLAG_URG = (1 << 5),
    /* set tcp packet with Urg flag */
    TCP_FLAG_ECE = (1 << 6),
    /* set tcp packet with ECE flag */
    TCP_FLAG_CWR = (1 << 7),
    /* set tcp packet with CQE flag */
};

struct ether_hdr
{
    uint8_t d_addr_bytes[ETHER_ADDR_LEN]; /* Destination addr bytes in tx order */
    uint8_t s_addr_bytes[ETHER_ADDR_LEN]; /* Source addr bytes in tx order */
    uint16_t ether_type;                  /* Frame type */
} __attribute__((__packed__));

struct ipv4_hdr
{
    uint8_t version_ihl;      /* version and header length */
    uint8_t type_of_service;  /* type of service */
    uint16_t total_length;    /* length of packet */
    uint16_t packet_id;       /* packet ID */
    uint16_t fragment_offset; /* fragmentation offset */
    uint8_t time_to_live;     /* time to live */
    uint8_t next_proto_id;    /* protocol ID */
    uint16_t hdr_checksum;    /* header checksum */
    uint32_t src_addr;        /* source address */
    uint32_t dst_addr;        /* destination address */
} __attribute__((__packed__));

struct eth_ip
{
    struct ether_hdr l2_hdr; /* Ethernet header */
    struct ipv4_hdr l3_hdr;  /* IP header */
} __attribute__((__packed__));

struct tcp_hdr
{
    uint16_t src_port; /* TCP source port */
    uint16_t dst_port; /* TCP destination port */
    uint32_t sent_seq; /* TX data sequence number */
    uint32_t recv_ack; /* RX data acknowledgment sequence number */
    uint8_t dt_off;    /* Data offset */
    uint8_t tcp_flags; /* TCP flags */
    uint16_t rx_win;   /* RX flow control window */
    uint16_t cksum;    /* TCP checksum */
    uint16_t tcp_urp;  /* TCP urgent pointer, if any */
} __attribute__((__packed__));

struct eth_ip_tcp_hdr
{
    struct ether_hdr l2_hdr; /* Ethernet header */
    struct ipv4_hdr l3_hdr;  /* IP header */
    struct tcp_hdr l4_hdr;   /* TCP header */
} __attribute__((__packed__));

struct udp_hdr
{
    uint16_t src_port;    /* UDP source port */
    uint16_t dst_port;    /* UDP destination port */
    uint16_t dgram_len;   /* UDP datagram length */
    uint16_t dgram_cksum; /* UDP datagram checksum */
} __attribute__((__packed__));

struct eth_ip_udp_hdr
{
    struct ether_hdr l2_hdr; /* Ethernet header */
    struct ipv4_hdr l3_hdr;  /* IP header */
    struct udp_hdr l4_hdr;   /* UDP header */
} __attribute__((__packed__));

// NOLINTEND(modernize-avoid-c-arrays)
// NOLINTEND(readability-identifier-naming)

#define TCP_HDR_SIZE sizeof(struct eth_ip_tcp_hdr)
#define UDP_HDR_SIZE sizeof(struct eth_ip_udp_hdr)

__device__ __inline__ int raw_to_tcp(const uintptr_t buf_addr, struct eth_ip_tcp_hdr** hdr, uint8_t** packet_data)
{
    (*hdr)         = (struct eth_ip_tcp_hdr*)buf_addr;
    (*packet_data) = (uint8_t*)(buf_addr + sizeof(struct ether_hdr) + sizeof(struct ipv4_hdr) +
                                (((*hdr)->l4_hdr.dt_off >> 4) * sizeof(int)));

    return 0;
}

__device__ __inline__ int raw_to_udp(const uintptr_t buf_addr, struct eth_ip_udp_hdr** hdr, uint8_t** packet_data)
{
    (*hdr)         = (struct eth_ip_udp_hdr*)buf_addr;
    (*packet_data) = (uint8_t*)(buf_addr + sizeof(struct eth_ip_udp_hdr));

    return 0;
}

__device__ __forceinline__ uint8_t gpu_ipv4_hdr_len(const struct ipv4_hdr& packet_l3)
{
    return (uint8_t)((packet_l3.version_ihl & RTE_IPV4_HDR_IHL_MASK) * RTE_IPV4_IHL_MULTIPLIER);
};

__device__ __forceinline__ uint32_t get_packet_size(ipv4_hdr& packet_l3)
{
    return static_cast<int32_t>(BYTE_SWAP16(packet_l3.total_length));
}

__device__ __forceinline__ int32_t get_payload_tcp_size(ipv4_hdr& packet_l3, tcp_hdr& packet_l4)
{
    auto packet_size       = get_packet_size(packet_l3);
    auto ip_header_length  = gpu_ipv4_hdr_len(packet_l3);
    auto tcp_header_length = static_cast<int32_t>(packet_l4.dt_off >> 4) * sizeof(int32_t);
    auto payload_size      = packet_size - ip_header_length - tcp_header_length;

    return payload_size;
}

__device__ __forceinline__ int32_t get_payload_udp_size(ipv4_hdr& packet_l3, udp_hdr& packet_l4)
{
    auto packet_size      = get_packet_size(packet_l3);
    auto ip_header_length = gpu_ipv4_hdr_len(packet_l3);
    auto payload_size     = packet_size - ip_header_length - sizeof(struct udp_hdr);

    return payload_size;
}

__device__ __forceinline__ uint32_t ip_to_int32(uint32_t address)
{
    return (address & 0x000000ff) << 24 | (address & 0x0000ff00) << 8 | (address & 0x00ff0000) >> 8 |
           (address & 0xff000000) >> 24;
}
