# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cupy as cp


def to_periodogram(signal):
    """
    Returns periodogram of signal for finding frequencies that have high energy.

    :param signal: signal (time domain)
    :type signal: cudf.Series
    :return: CuPy array representing periodogram
    :rtype: cupy.ndarray
    """

    # convert cudf series to cupy array
    signal_cp = cp.fromDlpack(signal.to_dlpack())

    # standardize the signal
    signal_cp_std = (signal_cp - cp.mean(signal_cp)) / cp.std(signal_cp)

    # take fourier transform of signal
    fft_data = cp.fft.fft(signal_cp_std)

    # create periodogram
    prdg = (1 / len(signal)) * ((cp.absolute(fft_data))**2)

    return prdg


def filter_periodogram(prdg, p_value):
    """
    Select important frequencies by filtering periodogram by p-value. Filtered out frequencies are set to zero.

    :param prdg: periodogram to be filtered
    :type prdg: cudf.Series
    :param p_value: p-value to filter by
    :type p_value: float
    :return: CuPy array representing periodogram
    :rtype: cupy.ndarray
    """

    filtered_prdg = cp.copy(prdg)
    filtered_prdg[filtered_prdg < (cp.mean(filtered_prdg) * (-1) * (cp.log(p_value)))] = 0

    return filtered_prdg


def to_time_domain(prdg):
    """
    Convert the signal back to time domain.

    :param prdg: periodogram (frequency domain)
    :type prdg: cupy.ndarray
    :return: CuPy array representing reconstructed signal
    :rtype: cupy.ndarray
    """

    acf = cp.abs(cp.fft.ifft(prdg))

    return acf
