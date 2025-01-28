# Copyright (c) 2023-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cudf

from morpheus.parsers import ip


def test_ip_to_int():
    input_df = cudf.Series(["5.79.97.178", "94.130.74.45"])
    expected = cudf.Series([89088434, 1585596973], dtype=cudf.api.types.dtype("uint32"))
    actual = ip.ip_to_int(input_df)
    assert actual.equals(expected)


def test_int_to_ip():
    input_df = cudf.Series([89088434, 1585596973], dtype=cudf.api.types.dtype("uint32"))
    expected = cudf.Series(["5.79.97.178", "94.130.74.45"])
    actual = ip.int_to_ip(input_df)
    assert actual.equals(expected)


def test_is_ip():
    input_df = cudf.Series(["5.79.97.178", "1.2.3.4", "5", "5.79", "5.79.97", "5.79.97.178.100"])
    expected = cudf.Series([True, True, False, False, False, False])
    actual = ip.is_ip(input_df)
    assert actual.equals(expected)


def test_is_reserved():
    input_df = cudf.Series(["240.0.0.0", "255.255.255.255", "5.79.97.178"])
    expected = cudf.Series([True, True, False])
    actual = ip.is_reserved(input_df)
    assert actual.equals(expected)


def test_is_loopback():
    input_df = cudf.Series(["127.0.0.1", "5.79.97.178"])
    expected = cudf.Series([True, False])
    actual = ip.is_loopback(input_df)
    assert actual.equals(expected)


def test_is_link_local():
    input_df = cudf.Series(["169.254.0.0", "5.79.97.178"])
    expected = cudf.Series([True, False])
    actual = ip.is_link_local(input_df)
    assert actual.equals(expected)


def test_is_unspecified():
    input_df = cudf.Series(["0.0.0.0", "5.79.97.178"])
    expected = cudf.Series([True, False])
    actual = ip.is_unspecified(input_df)
    assert actual.equals(expected)


def test_is_multicast():
    input_df = cudf.Series(["224.0.0.0", "239.255.255.255", "5.79.97.178"])
    expected = cudf.Series([True, True, False])
    actual = ip.is_multicast(input_df)
    assert actual.equals(expected)


def test_is_private():
    input_df = cudf.Series(["0.0.0.0", "5.79.97.178"])
    expected = cudf.Series([True, False])
    actual = ip.is_private(input_df)
    assert actual.equals(expected)


def test_is_global():
    input_df = cudf.Series(["0.0.0.0", "5.79.97.178"])
    expected = cudf.Series([False, True])
    actual = ip.is_global(input_df)
    assert actual.equals(expected)


def test_netmask():
    input_df = cudf.Series(["5.79.97.178", "94.130.74.45"])
    expected = cudf.Series(["255.255.128.0", "255.255.128.0"])
    actual = ip.netmask(input_df, 17)
    assert actual.equals(expected)


def test_hostmask():
    input_df = cudf.Series(["5.79.97.178", "94.130.74.45"])
    expected = cudf.Series(["0.0.127.255", "0.0.127.255"])
    actual = ip.hostmask(input_df, 17)
    assert actual.equals(expected)


def test_mask():
    input_ips = cudf.Series(["5.79.97.178", "94.130.74.45"])
    input_masks = cudf.Series(["255.255.128.0", "255.255.128.0"])
    expected = cudf.Series(["5.79.0.0", "94.130.0.0"])
    actual = ip.mask(input_ips, input_masks)
    assert actual.equals(expected)
