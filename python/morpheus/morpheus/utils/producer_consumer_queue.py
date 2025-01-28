# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import queue
import threading
import typing
from time import time

_T = typing.TypeVar("_T")


class Closed(Exception):
    'Exception raised when the queue is closed'
    pass


class ProducerConsumerQueue(queue.Queue, typing.Generic[_T]):
    """
    Custom queue.Queue implementation which supports closing and uses recursive locks

    Parameters
    ----------
    maxsize : int
         Maximum size of queue. If maxsize is <= 0, the queue size is infinite.
    """

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize=maxsize)

        # Use a recursive lock here to prevent reentrant deadlocks
        self.mutex = threading.RLock()

        self.not_empty = threading.Condition(self.mutex)
        self.not_full = threading.Condition(self.mutex)
        self.all_tasks_done = threading.Condition(self.mutex)

        self._is_closed = False

    def join(self):
        """
        Blocks until the queue has been closed and all tasks are completed
        """
        with self.all_tasks_done:
            while not self._is_closed and self.unfinished_tasks:
                self.all_tasks_done.wait()

    def put(self, item: _T, block: bool = True, timeout: typing.Optional[float] = None) -> None:
        """
        Put an item into the back of the queue. When `block` is `True` and the queue is full it will block up to
        `timeout` seconds, raising a  `queue.Full` when either `block` is `False` or the `timeout` has exceeded. A
        `Closed` exception is raised if the queue is closed.
        """
        with self.not_full:
            if self.maxsize > 0:
                if not block:
                    if self._qsize() >= self.maxsize and not self._is_closed:
                        raise queue.Full  # @IgnoreException
                elif timeout is None:
                    while self._qsize() >= self.maxsize and not self._is_closed:
                        self.not_full.wait()
                elif timeout < 0:
                    raise ValueError("'timeout' must be a non-negative number")
                else:
                    endtime = time() + timeout
                    while self._qsize() >= self.maxsize and not self._is_closed:
                        remaining = endtime - time()
                        if remaining <= 0.0:
                            raise queue.Full  # @IgnoreException
                        self.not_full.wait(remaining)

            if (self._is_closed):
                raise Closed  # @IgnoreException

            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()

    def get(self, block: bool = True, timeout: typing.Optional[float] = None) -> _T:
        """
        Remove and return an item from the front of the queue. When `block` is `True` and the queue is empty it will
        block up to `timeout` seconts, raising a  `queue.Empty` when either `block` is `False` or the `timeout` has
        exceeded. A `Closed` exception is raised if the queue is closed.
        """
        with self.not_empty:
            if not block:
                if not self._qsize() and not self._is_closed:
                    raise queue.Empty  # @IgnoreException
            elif timeout is None:
                while not self._qsize() and not self._is_closed:
                    self.not_empty.wait()
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                endtime = time() + timeout
                while not self._qsize() and not self._is_closed:
                    remaining = endtime - time()
                    if remaining <= 0.0:
                        raise queue.Empty  # @IgnoreException
                    self.not_empty.wait(remaining)

            if (self._is_closed and not self._qsize()):
                raise Closed  # @IgnoreException

            item = self._get()
            self.not_full.notify()
            return item

    def close(self):
        """Close the queue."""
        with self.mutex:
            if (not self._is_closed):
                self._is_closed = True
                self.not_full.notify_all()
                self.not_empty.notify_all()
                self.all_tasks_done.notify_all()

    def is_closed(self) -> bool:
        """Check if the queue is closed."""
        with self.mutex:
            return self._is_closed


class AsyncIOProducerConsumerQueue(asyncio.Queue, typing.Generic[_T]):
    """
    Custom queue.Queue implementation which supports closing and uses recursive locks
    """

    def __init__(self, maxsize=0) -> None:
        super().__init__(maxsize=maxsize)

        self._closed = asyncio.Event()
        self._is_closed = False

    async def join(self):
        """Block until all items in the queue have been gotten and processed.

        The count of unfinished tasks goes up whenever an item is added to the
        queue. The count goes down whenever a consumer calls task_done() to
        indicate that the item was retrieved and all work on it is complete.
        When the count of unfinished tasks drops to zero, join() unblocks.
        """

        # First wait for the closed flag to be set
        await self._closed.wait()

        if self._unfinished_tasks > 0:
            await self._finished.wait()

    async def put(self, item):
        """Put an item into the queue.

        Put an item into the queue. If the queue is full, wait until a free
        slot is available before adding item.
        """
        while self.full() and not self._is_closed:
            putter = self._get_loop().create_future()
            self._putters.append(putter)
            try:
                await putter
            except Exception:
                putter.cancel()  # Just in case putter is not done yet.
                try:
                    # Clean self._putters from canceled putters.
                    self._putters.remove(putter)
                except ValueError:
                    # The putter could be removed from self._putters by a
                    # previous get_nowait call.
                    pass
                if not self.full() and not putter.cancelled():
                    # We were woken up by get_nowait(), but can't take
                    # the call.  Wake up the next in line.
                    self._wakeup_next(self._putters)
                raise

        if (self._is_closed):
            raise Closed  # @IgnoreException

        return self.put_nowait(item)

    async def get(self) -> _T:
        """Remove and return an item from the queue.

        If queue is empty, wait until an item is available.
        """
        while self.empty() and not self._is_closed:
            getter = self._get_loop().create_future()
            self._getters.append(getter)
            try:
                await getter
            except Exception:
                getter.cancel()  # Just in case getter is not done yet.
                try:
                    # Clean self._getters from canceled getters.
                    self._getters.remove(getter)
                except ValueError:
                    # The getter could be removed from self._getters by a
                    # previous put_nowait call.
                    pass
                if not self.empty() and not getter.cancelled():
                    # We were woken up by put_nowait(), but can't take
                    # the call.  Wake up the next in line.
                    self._wakeup_next(self._getters)
                raise

        if (self.empty() and self._is_closed):
            raise Closed  # @IgnoreException

        return self.get_nowait()

    async def close(self):
        """Close the queue."""
        if (not self._is_closed):
            self._is_closed = True

            # Hit the flag
            self._closed.set()

            self._wakeup_next(self._putters)
            self._wakeup_next(self._getters)

    def is_closed(self) -> bool:
        """Check if the queue is closed."""
        return self._is_closed
