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

import logging
import multiprocessing as mp
import threading
from decimal import Decimal
from fractions import Fraction

import pytest

from morpheus.utils.shared_process_pool import PoolStatus
from morpheus.utils.shared_process_pool import SharedProcessPool

logger = logging.getLogger(__name__)

# This test has issues with joining processes when testing with pytest `-s` option. Run pytest without `-s` flag


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown(shared_process_pool_setup_and_teardown):  # pylint: disable=unused-argument
    pass


@pytest.fixture(name="shared_process_pool", scope="function")
def shared_process_pool_fixture():

    pool = SharedProcessPool()
    pool.wait_until_ready()
    yield pool

    # Stop and reset the pool after each test
    pool.stop()
    pool.join()
    pool.reset()


def _add_task(x, y):
    return x + y


def _blocked_until_signaled_task(q: mp.Queue):
    return q.get()


def _function_raises_exception():
    raise RuntimeError("Exception is raised in the process.")


def _function_returns_unserializable_result():
    return threading.Lock()


def _arbitrary_function(*args, **kwargs):
    return args, kwargs


def _check_pool_stage_settings(pool: SharedProcessPool, stage_name: str, usage: float):

    assert pool._stage_usage.get(stage_name) == usage
    assert stage_name in pool._stage_semaphores
    assert stage_name in pool._task_queues


def test_singleton():

    pool_1 = SharedProcessPool()
    pool_2 = SharedProcessPool()

    assert pool_1 is pool_2


@pytest.mark.slow
def test_pool_status(shared_process_pool):

    pool = shared_process_pool
    assert pool.status == PoolStatus.RUNNING

    pool.set_usage("test_stage", 0.5)

    pool.stop()
    pool.join()
    assert pool.status == PoolStatus.SHUTDOWN

    # After pool.start(), the pool should have the same status as before stopping
    pool.start()
    pool.wait_until_ready()
    assert pool.status == PoolStatus.RUNNING
    assert pool._total_usage == 0.5
    _check_pool_stage_settings(pool, "test_stage", 0.5)

    pool.stop()
    pool.join()
    assert pool.status == PoolStatus.SHUTDOWN

    # After pool.reset(), the pool should reset all the status
    pool.reset()
    pool.wait_until_ready()
    assert pool.status == PoolStatus.RUNNING
    assert pool._total_usage == 0.0
    assert not pool._stage_usage
    assert not pool._stage_semaphores
    assert not pool._task_queues


@pytest.mark.slow
@pytest.mark.parametrize(
    "a, b, expected",
    [
        (1, 2, 3),  # Integers
        (complex(1, 2), complex(3, 4), complex(4, 6)),  # Complex numbers
        (Decimal('1.1'), Decimal('2.2'), Decimal('3.3')),  # Decimal numbers
        (Fraction(1, 2), Fraction(1, 3), Fraction(5, 6)),  # Fractions
        ("Hello, ", "world!", "Hello, world!"),  # Strings
        ([1, 2, 3], [4, 5, 6], [1, 2, 3, 4, 5, 6]),  # Lists
        ((1, 2, 3), (4, 5, 6), (1, 2, 3, 4, 5, 6)),  # Tuples
    ])
def test_submit_single_task(shared_process_pool, a, b, expected):

    pool = shared_process_pool
    pool.set_usage("test_stage", 0.5)

    task = pool.submit_task("test_stage", _add_task, a, b)
    assert task.result() == expected

    task = pool.submit_task("test_stage", _add_task, x=a, y=b)
    assert task.result() == expected

    task = pool.submit_task("test_stage", _add_task, a, y=b)
    assert task.result() == expected

    pool.stop()

    # After the pool is stopped, it should not accept any new tasks
    with pytest.raises(RuntimeError):
        pool.submit_task("test_stage", _add_task, 10, 20)


@pytest.mark.slow
def test_submit_invalid_tasks(shared_process_pool):

    pool = shared_process_pool

    # submit_task() should raise ValueError if the stage does not exist
    with pytest.raises(ValueError):
        pool.submit_task("stage_does_not_exist", _add_task, 10, 20)

    pool.set_usage("test_stage", 0.5)

    # if the function raises exception, the task can be submitted and the exception will be raised when calling result()
    task = pool.submit_task("test_stage", _function_raises_exception)
    with pytest.raises(RuntimeError):
        task.result()

    # if the function returns unserializable result, the task can be submitted and the exception will be raised
    # when calling result()
    task = pool.submit_task("test_stage", _function_returns_unserializable_result)
    with pytest.raises(TypeError):
        task.result()

    # Function with unserializable arguments cannot be submitted to the pool
    with pytest.raises(TypeError):
        pool.submit_task("test_stage", _arbitrary_function, threading.Lock())


@pytest.mark.slow
@pytest.mark.parametrize(
    "a, b, expected",
    [
        (1, 2, 3),  # Integers
        (complex(1, 2), complex(3, 4), complex(4, 6)),  # Complex numbers
        (Decimal('1.1'), Decimal('2.2'), Decimal('3.3')),  # Decimal numbers
        (Fraction(1, 2), Fraction(1, 3), Fraction(5, 6)),  # Fractions
        ("Hello, ", "world!", "Hello, world!"),  # Strings
        ([1, 2, 3], [4, 5, 6], [1, 2, 3, 4, 5, 6]),  # Lists
        ((1, 2, 3), (4, 5, 6), (1, 2, 3, 4, 5, 6)),  # Tuples
    ])
def test_submit_multiple_tasks(shared_process_pool, a, b, expected):

    pool = shared_process_pool
    pool.set_usage("test_stage", 0.5)

    num_tasks = 10
    tasks = []
    for _ in range(num_tasks):
        tasks.append(pool.submit_task("test_stage", _add_task, a, b))

    for future in tasks:
        assert future.result() == expected


@pytest.mark.slow
def test_set_usage(shared_process_pool):

    pool = shared_process_pool

    pool.set_usage("test_stage_1", 0.5)
    assert pool._total_usage == 0.5
    _check_pool_stage_settings(pool, "test_stage_1", 0.5)

    pool.set_usage("test_stage_2", 0.3)
    assert pool._total_usage == 0.8
    _check_pool_stage_settings(pool, "test_stage_2", 0.3)

    # valid update to the usage of an existing stage
    pool.set_usage("test_stage_1", 0.6)
    assert pool._total_usage == 0.9
    _check_pool_stage_settings(pool, "test_stage_1", 0.6)

    # invalid update to the usage of an existing stage, exceeding the total usage limit
    with pytest.raises(ValueError):
        pool.set_usage("test_stage_1", 0.8)

    # adding a new stage usage, exceeding the total usage limit
    with pytest.raises(ValueError):
        pool.set_usage("test_stage_3", 0.2)

    with pytest.raises(ValueError):
        pool.set_usage("test_stage_1", 1.1)

    with pytest.raises(ValueError):
        pool.set_usage("test_stage_1", -0.1)

    # invalid settings should not change the pool status
    _check_pool_stage_settings(pool, "test_stage_1", 0.6)
    assert pool._total_usage == 0.9


@pytest.mark.slow
def test_task_completion_with_early_stop(shared_process_pool):

    pool = shared_process_pool
    pool.set_usage("test_stage_1", 0.1)
    pool.set_usage("test_stage_2", 0.3)
    pool.set_usage("test_stage_3", 0.5)

    manager = mp.Manager()
    queue = manager.Queue()

    tasks = []

    task_num = 10

    for _ in range(task_num):
        tasks.append(pool.submit_task("test_stage_1", _blocked_until_signaled_task, queue))
        tasks.append(pool.submit_task("test_stage_2", _blocked_until_signaled_task, queue))
        tasks.append(pool.submit_task("test_stage_3", _blocked_until_signaled_task, queue))

    pool.stop()

    # No tasks have been completed since they have not been signaled yet
    for task in tasks:
        assert not task.done()

    for i in range(len(tasks)):
        queue.put(i)

    pool.join()

    # all tasks should be completed before the pool is shutdown
    assert len(tasks) == 3 * task_num
    for task in tasks:
        assert task.done()
