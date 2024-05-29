from datetime import datetime, timedelta
from threading import Thread, Lock
from typing import Callable, TypeVar, Generic

T = TypeVar('T')

class DebounceQueue(Generic[T]):

    _lock: Lock
    _items: list[T]
    _oldest_time: datetime
    _latest_time: datetime
    _callback: Callable[[list[T]], None]
    _now: Callable[[],datetime]
    _min_delay: timedelta
    _max_delay: timedelta


    def __init__(self,
                 callback: Callable[[list[T]], None],
                 now: Callable[[],datetime]=datetime.now,
                 min_delay: timedelta=timedelta(seconds=1),
                 max_delay: timedelta=timedelta(seconds=5)
        ):
        self._lock = Lock()
        self._items = []
        self._oldest_time = None
        self._latest_time = None
        self._callback = callback
        self._now = now
        self._min_delay = min_delay
        self._max_delay = max_delay


    def queue(self, item: T):
        with self._lock:
            now = self._now()
            if self._oldest_time is None:
                self._oldest_time = now
            self._latest_time = now
            self._items.append(item)


    def step(self):
        with self._lock:
            if self._oldest_time is None:
                return # nothing has been queued.
            
            now = self._now()
            
            if (now - self._latest_time) > self._min_delay:
                self._flush()
                return

            if (now - self._oldest_time) > self._max_delay:
                self._flush()
                return
    

    def flush(self):
        with self._lock:
            self._flush()


    def _flush(self):
        if self._oldest_time is None:
            return # nothing has been queued.

        self._oldest_time = None
        self._latest_time = None
        self._callback(self._items)
        self._items = []


class DebounceRunner():

    _cancelled: bool
    _thread: Thread
    _queue: DebounceQueue
    _step_delay: float


    def __init__(self, queue: DebounceQueue, step_delay: float = 0.1):
        self._thread = Thread(target=self._run)
        self._queue = queue
        self._step_delay = step_delay
        self._cancelled = False


    def start(self):
        self._thread.start()


    def stop(self):
        if self._thread.is_alive:
            self._cancelled = True
            self._thread.join()
            self._queue.flush()


    def __del__(self):
        self.stop()


    def _run(self):
        import time
        while not self._cancelled:
            self._queue.step()
            time.sleep(self._step_delay)
