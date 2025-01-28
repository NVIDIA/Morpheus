#!/usr/bin/python
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os

import click
from scapy.all import IP  # pylint: disable=no-name-in-module
from scapy.all import TCP
from scapy.all import UDP  # pylint: disable=no-name-in-module
from scapy.all import RandShort
from scapy.all import Raw
from scapy.all import send

DEFAULT_DPORT = 5001
MORPHEUS_ROOT = os.environ['MORPHEUS_ROOT']


def get_data(input_glob: str) -> list[str]:
    data = []
    for file in glob.glob(input_glob):
        with open(file, 'r', encoding='utf-8') as fh:
            while True:
                content = fh.read(1024)
                if not content:
                    break

                data.append(content)

    return data


def send_data(data: list[str],
              dst_ip: str,
              dport: int = DEFAULT_DPORT,
              iface: str | None = None,
              src_ip: str | None = None,
              sport: int | None = None,
              net_type: str = 'UDP'):
    if net_type == 'UDP':
        net_type_cls = UDP
    else:
        net_type_cls = TCP

    if sport is None:
        sport = RandShort()

    ip_kwargs = {"dst": dst_ip}
    if src_ip is not None:
        ip_kwargs["src"] = src_ip

    packets = [
        IP(**ip_kwargs) / net_type_cls(sport=sport, dport=dport) / Raw(load=content.encode('utf-8')) for content in data
    ]

    send_kwargs = {}
    if iface is not None:
        send_kwargs["iface"] = iface

    send(packets, **send_kwargs)


@click.command()
@click.option("--iface", help="Ethernet device to use, useful for systems with multiple NICs", required=False)
@click.option("--src_ip", help="Source IP to send from, useful for systems with multiple IPs", required=False)
@click.option("--dst_ip", help="Destination IP to send to", required=True)
@click.option("--dport", help="Destination port", type=int, default=DEFAULT_DPORT)
@click.option("--sport",
              help="Source port, if undefined a random port will be used",
              type=int,
              default=None,
              required=False)
@click.option("--net_type", type=click.Choice(['TCP', 'UDP'], case_sensitive=False), default='UDP')
@click.option("--input_data_glob",
              type=str,
              default=os.path.join(MORPHEUS_ROOT, 'examples/doca/vdb_realtime/sender/dataset/*.txt'),
              help="Input filepath glob pattenr matching the data to send.")
def main(iface: str | None,
         src_ip: str | None,
         dst_ip: str,
         dport: int,
         sport: int | None,
         net_type: str,
         input_data_glob: str):
    data = get_data(input_data_glob)
    send_data(data=data, dst_ip=dst_ip, dport=dport, iface=iface, src_ip=src_ip, sport=sport, net_type=net_type)


if __name__ == "__main__":
    main()
