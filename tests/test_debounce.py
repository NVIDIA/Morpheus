from datetime import datetime, timedelta
import pytest

from morpheus.utils.debounce import DebounceQueue, DebounceRunner


def test_slow():

    import time

    flushed = []

    def flush(keys):
        for key in keys:
            flushed.append(key)

    queue = DebounceQueue(flush)
    runner = DebounceRunner(queue)

    runner.start()

    queue.queue("a")
    queue.queue("b")

    time.sleep(2)

    assert len(flushed) == 2

    queue.queue("c")
    queue.queue("d")

    runner.stop()

    assert len(flushed) == 4

    with pytest.raises(RuntimeError):
        runner.start()


def test_flush():
    time = datetime(2024, 1, 1)

    def get_time():
        return time

    flushed = {}

    def flush(keys):
        for item in keys:
            flushed[item] = time

    batcher = DebounceQueue(flush, now=get_time)

    batcher.queue("a")
    batcher.queue("b")

    assert len(flushed) == 0

    batcher.flush()

    assert len(flushed) == 2
    assert flushed["a"] == time
    assert flushed["b"] == time

def test_step_no_primer():
    time = datetime(2024, 1, 1)

    def get_time():
        return time

    flushed = {}

    def flush(keys):
        for item in keys:
            flushed[item] = time

    batcher = DebounceQueue(flush, now=get_time)

    time += timedelta(days=1)

    batcher.queue("a")
    batcher.step()

    assert len(flushed) == 0

    time += timedelta(days=1)

    batcher.step()

    assert len(flushed) == 1


def test_step_min_delay():
    time = datetime(2024, 1, 1)

    def get_time():
        return time

    flushed = {}

    def flush(keys):
        for item in keys:
            flushed[item] = time

    batcher = DebounceQueue(flush, now=get_time)

    batcher.queue("a")
    time += timedelta(seconds=1)

    batcher.step()

    assert len(flushed) == 0

    batcher.queue("b")
    time += timedelta(seconds=1)

    batcher.step()

    assert len(flushed) == 0

    time += timedelta(seconds=1)

    batcher.step()

    assert len(flushed) == 2


def test_step_max_delay():

    time = datetime(2024, 1, 1)
    time_start = time

    def get_time():
        return time

    flushed = {}

    def flush(keys):
        for item in keys:
            flushed[item] = time

    batcher = DebounceQueue(flush, now=get_time)

    for i in range(10):
        batcher.queue(i)
        time += timedelta(seconds=1)
        batcher.step()

    assert len(flushed) == 6
    assert flushed[0] == time_start + timedelta(seconds=6)
    assert flushed[1] == time_start + timedelta(seconds=6)
    assert flushed[2] == time_start + timedelta(seconds=6)
    assert flushed[3] == time_start + timedelta(seconds=6)
    assert flushed[4] == time_start + timedelta(seconds=6)
    assert flushed[5] == time_start + timedelta(seconds=6)

    time += timedelta(seconds=10)

    batcher.step()

    assert len(flushed) == 10

    assert flushed[6] == time_start + timedelta(seconds=20)
    assert flushed[7] == time_start + timedelta(seconds=20)
    assert flushed[8] == time_start + timedelta(seconds=20)
    assert flushed[9] == time_start + timedelta(seconds=20)
