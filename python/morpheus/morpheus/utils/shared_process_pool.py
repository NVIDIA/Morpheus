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
import math
import multiprocessing as mp
import os
import queue
import threading
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


class PoolStatus(Enum):
    INITIALIZING = 0
    RUNNING = 1
    STOPPED = 2
    SHUTDOWN = 3


class SimpleFuture:

    def __init__(self, manager):
        self._result = manager.Value("i", None)
        self._exception = manager.Value("i", None)
        self._done = manager.Event()

    def set_result(self, result):
        self._result.value = result
        self._done.set()

    def set_exception(self, exception):
        self._exception.value = exception
        self._done.set()

    def result(self):
        self._done.wait()
        if self._exception.value is not None:
            raise self._exception.value
        return self._result.value

    def done(self):
        return self._done.is_set()


class Task(SimpleFuture):

    def __init__(self, manager, process_fn: Callable, args, kwargs):
        super().__init__(manager)
        self._process_fn = process_fn
        self._args = args
        self._kwargs = kwargs

    @property
    def process_fn(self):
        return self._process_fn

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs


class CancellationToken:

    def __init__(self, manager):
        self._cancel_event = manager.Event()

    def cancel(self):
        self._cancel_event.set()

    def is_cancelled(self):
        return self._cancel_event.is_set()


# pylint: disable=W0201
class SharedProcessPool:

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """
        Singleton pattern for SharedProcessPool.

        Returns
        -------
        cls._instance : SharedProcessPool
            The SharedProcessPool instance.

        Raises
        ------
        RuntimeError
            If SharedProcessPool() is called when the instance already exists but not running
        """

        with cls._lock:
            if cls._instance is None:
                logger.debug("SharedProcessPool.__new__: Creating a new instance...")
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
                logger.debug("SharedProcessPool.__new__: SharedProcessPool has been initialized.")

            else:
                logger.debug("SharedProcessPool.__new__: instance already exists.")

        return cls._instance

    def _initialize(self):
        self._status = PoolStatus.INITIALIZING

        cpu_usage = os.environ.get("MORPHEUS_SHARED_PROCESS_POOL_CPU_USAGE", None)
        if cpu_usage is not None:
            cpu_usage = float(cpu_usage)
            if not 0 <= cpu_usage <= 1:
                raise ValueError("CPU usage must be between 0 and 1.")
        else:
            cpu_usage = 0.5
        self._total_max_workers = math.floor(max(1, len(os.sched_getaffinity(0)) * cpu_usage))
        self._processes = []

        self._context = mp.get_context("forkserver")
        self._manager = self._context.Manager()
        self._task_queues = self._manager.dict()
        self._stage_semaphores = self._manager.dict()
        self._total_usage = 0.0
        self._stage_usage = {}

        self._cancellation_token = CancellationToken(self._manager)
        self._launch_condition = threading.Condition()
        self._join_condition = threading.Condition()

        self.start()

    def _launch_workers(self):
        for i in range(self.total_max_workers):
            process = self._context.Process(target=self._worker,
                                            args=(self._cancellation_token, self._task_queues, self._stage_semaphores))
            process.start()
            self._processes.append(process)
            logger.debug("SharedProcessPool._lanch_workers(): Process %s/%s has been started.",
                         i + 1,
                         self.total_max_workers)

        with self._launch_condition:
            self._launch_condition.notify_all()

        self._status = PoolStatus.RUNNING

    @property
    def total_max_workers(self):
        return self._total_max_workers

    @property
    def status(self) -> PoolStatus:
        return self._status

    @staticmethod
    def _worker(cancellation_token, task_queues, stage_semaphores):
        logger.debug("SharedProcessPool._worker: Worker process %s has been started.", os.getpid())

        while True:
            if cancellation_token.is_cancelled():
                logger.debug("SharedProcessPool._worker: Worker process %s has terminated the worker loop.",
                             os.getpid())
                return

            for stage_name, task_queue in task_queues.items():
                semaphore = stage_semaphores[stage_name]

                if not semaphore.acquire(blocking=False):
                    # Stage has reached the limitation of processes
                    continue

                try:
                    task = task_queue.get(timeout=0.1)
                except queue.Empty:
                    semaphore.release()
                    continue

                if task is None:
                    logger.debug("SharedProcessPool._worker: Worker process %s has received a None task.", os.getpid())
                    semaphore.release()
                    continue

                process_fn = task.process_fn
                args = task.args
                kwargs = task.kwargs

                try:
                    result = process_fn(*args, **kwargs)
                    task.set_result(result)
                except Exception as e:
                    task.set_exception(e)

                task_queue.task_done()

                semaphore.release()

    def _join_process_pool(self):
        for task_queue in self._task_queues.values():
            task_queue.join()

        self._cancellation_token.cancel()
        for i, p in enumerate(self._processes):
            p.join()
            logger.debug("Process %s/%s has been joined.", i + 1, len(self._processes))

        with self._join_condition:
            self._join_condition.notify_all()

    def submit_task(self, stage_name: str, process_fn: Callable, *args, **kwargs) -> Task:
        """
        Submit a task to the SharedProcessPool.

        Parameters
        ----------
        stage_name : str
            The unique name of the stage.
        process_fn : Callable
            The function to be executed in the process pool.
        args : Any
            Arbitrary arguments for the process_fn.
        kwargs : Any
            Arbitrary keyword arguments for the process_fn.

        Returns
        -------
        Task
            The task object that includes the result of the process_fn.

        Raises
        ------
        RuntimeError
            If the SharedProcessPool is not running.
        ValueError
            If the stage_name has not been set in the SharedProcessPool.
        """
        if self._status != PoolStatus.RUNNING:
            raise RuntimeError("Cannot submit a task to a SharedProcessPool that is not running.")

        if stage_name not in self._stage_usage:
            raise ValueError(f"Stage {stage_name} has not been set in SharedProcessPool.")

        task = Task(self._manager, process_fn, args, kwargs)
        self._task_queues[stage_name].put(task)

        return task

    def set_usage(self, stage_name: str, percentage: float):
        """
        Set the usage of the SharedProcessPool for a specific stage.

        Parameters
        ----------
        stage_name : str
            The unique name of the stage.
        percentage : float
            The percentage of the total workers that will be allocated to the stage, should be between 0 and 1.

        Raises
        ------
        RuntimeError
            If the SharedProcessPool is not running.
        ValueError
            If the percentage is not between 0 and 1 or the total usage is greater than 1.
        """
        if self._status != PoolStatus.RUNNING:
            raise RuntimeError("Cannot set usage to a SharedProcessPool that is not running.")

        if not 0 <= percentage <= 1:
            raise ValueError("Percentage must be between 0 and 1.")

        new_total_usage = self._total_usage - self._stage_usage.get(stage_name, 0.0) + percentage

        if new_total_usage > 1.0:
            raise ValueError("Total percentage cannot exceed 1.0.")

        self._stage_usage[stage_name] = percentage
        self._total_usage = new_total_usage

        allowed_processes_num = max(1, int(self._total_max_workers * percentage))
        self._stage_semaphores[stage_name] = self._manager.Semaphore(allowed_processes_num)

        if stage_name not in self._task_queues:
            self._task_queues[stage_name] = self._manager.Queue()

        logger.debug("SharedProcessPool.set_usage(): stage_usage: %s", self._stage_usage)
        logger.debug("SharedProcessPool.set_usage(): stage semaphores: %s", allowed_processes_num)

    def start(self):
        """
        Start the SharedProcessPool that is currently stopped and keep the settings before last shutdown.

        Raises
        ------
        RuntimeError
            If the SharedProcessPool is not shutdown.
        """
        if self._status == PoolStatus.RUNNING:
            logger.debug("SharedProcessPool.start(): process pool is already running.")
            return

        process_launcher = threading.Thread(target=self._launch_workers)
        process_launcher.start()
        process_launcher.join()

    def wait_until_ready(self, timeout: float | None = None):
        """
        Wait until the SharedProcessPool is running and ready to accept tasks.

        Parameters
        ----------
        timeout : _type_, optional
            timeout in seconds to wait for the SharedProcessPool to be ready, by default None.
            If None, it will wait indefinitely.

        Raises
        ------
        RuntimeError
            If the SharedProcessPool is not initializing or running.
        TimeoutError
            If has been waiting more than the timeout.
        """
        if self._status == PoolStatus.RUNNING:
            logger.debug("SharedProcessPool.wait_until_ready(): SharedProcessPool is already running.")
            return

        if self._status == PoolStatus.INITIALIZING:
            with self._launch_condition:
                launched = self._launch_condition.wait(timeout)
                if not launched:
                    raise TimeoutError("Time out.")
        else:
            raise RuntimeError("Cannot wait for a pool that is not initializing or running.")

        logger.debug("SharedProcessPool.wait_until_ready(): SharedProcessPool is ready.")

    def reset(self):
        """
        Clear all the previous settings and restart the SharedProcessPool.

        Raises
        ------
        RuntimeError
            If the SharedProcessPool is not already shut down.
        """
        if self._status != PoolStatus.SHUTDOWN:
            raise RuntimeError("Cannot reset a SharedProcessPool that is not already shut down.")

        self._initialize()

    def stop(self):
        """
        Stop receiving any new tasks.
        """
        if self._status not in (PoolStatus.RUNNING, PoolStatus.INITIALIZING):
            logger.debug("SharedProcessPool.stop(): Cannot stop a SharedProcessPool that is not running.")
            return

        # No new tasks will be accepted from this point
        self._status = PoolStatus.STOPPED

    def join(self, timeout: float | None = None):
        """
        Wait until the SharedProcessPool is terminated.

        Parameters
        ----------
        timeout : _type_, optional
            timeout in seconds to wait for the SharedProcessPool to be terminated, by default None.
            If None, it will wait indefinitely.

        Raises
        ------
        RuntimeError
            If is called on a SharedProcessPool that is not stopped.

        TimeoutError
            If has been waiting more than the timeout.
        """

        if self._status != PoolStatus.STOPPED:
            if self._status == PoolStatus.SHUTDOWN:
                logger.debug("SharedProcessPool.join(): process pool is already shut down.")
                return

            raise RuntimeError("Cannot join SharedProcessPool that is not stopped.")

        process_joiner = threading.Thread(target=self._join_process_pool)
        process_joiner.start()

        with self._join_condition:
            joined = self._join_condition.wait(timeout)
            if not joined:
                raise TimeoutError("time out.")

        process_joiner.join()

        self._status = PoolStatus.SHUTDOWN

        logger.debug("SharedProcessPool.join(): SharedProcessPool has been joined.")

    def terminate(self):
        """
        Terminate all processes and shutdown the SharedProcessPool immediately.
        """
        for i, p in enumerate(self._processes):
            p.terminate()
            logger.debug("Process %s/%s has been terminated.", i + 1, self._total_max_workers)

        logger.debug("SharedProcessPool.terminate(): SharedProcessPool has been terminated.")
        self._status = PoolStatus.SHUTDOWN

    def __del__(self):
        self.terminate()
