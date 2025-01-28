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

import queue

import pytest

from morpheus.common import FiberQueue
from morpheus.utils.producer_consumer_queue import Closed


@pytest.mark.parametrize("invalid_max_size", [0, 1, 3, 9])
def test_constructor_errors(invalid_max_size: int):
    with pytest.raises(ValueError):
        FiberQueue(invalid_max_size)


@pytest.mark.parametrize("invalid_max_size", [-1, -2, -4])
def test_constructor_type_error(invalid_max_size: int):
    """
    On the C++ side the `max_size` parameter is a `size_t`, any negative values will be caught by pybind11 and raised
    as a TypeError.
    """
    with pytest.raises(TypeError):
        FiberQueue(invalid_max_size)


@pytest.mark.parametrize("max_size", [2, 4, 8, 16])
def test_put_get(max_size: int):
    with FiberQueue(max_size) as data_queue:
        for i in range(max_size - 1):
            data_queue.put(item=i, block=False)

        for expected in range(max_size - 1):
            assert data_queue.get(block=False) == expected


@pytest.mark.parametrize("max_size", [2, 4, 8, 16])
def test_put_full_error(max_size: int):
    with FiberQueue(max_size) as data_queue:
        for i in range(max_size - 1):
            data_queue.put(item=i, block=False)

        with pytest.raises(queue.Full):
            data_queue.put(item=42, block=False)


@pytest.mark.parametrize("block", [True, False])
def test_get_empty_error(block: bool):
    data_queue = FiberQueue(8)

    with pytest.raises(queue.Empty):
        data_queue.get(block=block, timeout=0.1)


def test_closed_error():
    data_queue = FiberQueue(8)
    assert not data_queue.is_closed()

    data_queue.close()
    assert data_queue.is_closed()

    with pytest.raises(Closed):
        data_queue.get(block=False)

    with pytest.raises(Closed):
        data_queue.put(item=5, block=False)


def test_ctx_closed_error():
    with FiberQueue(8) as data_queue:
        assert not data_queue.is_closed()

    assert data_queue.is_closed()

    with pytest.raises(Closed):
        data_queue.get(block=False)

    with pytest.raises(Closed):
        data_queue.put(item=5, block=False)
