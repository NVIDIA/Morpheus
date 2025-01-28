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

import ctypes
import multiprocessing
import signal
import sys
import time


def _wrap(fn, i, args, error_queue):

    libc = ctypes.CDLL("libc.so.6")
    libc.prctl(1, signal.SIGINT)

    # Force import of libstdc++.so
    libstdcpp = ctypes.CDLL("libstdc++.so.6")

    try:
        fn(i, *args)
    except KeyboardInterrupt:
        pass  # SIGINT; Killed by parent, do nothing
    except Exception:
        # Propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put(traceback.format_exc())
        sys.exit(1)


def start_processes(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn'):
    """
    Ensure we are running with the correct version of libstdc++
    """
    mp = multiprocessing.get_context(start_method)
    error_queues = []
    processes = []
    for i in range(nprocs):
        error_queue = mp.SimpleQueue()
        process = mp.Process(
            target=_wrap,
            args=(fn, i, args, error_queue),
            daemon=daemon,
        )
        process.start()
        error_queues.append(error_queue)
        processes.append(process)

    from torch.multiprocessing.spawn import ProcessContext
    context = ProcessContext(processes, error_queues)
    if not join:
        return context

    # Loop on join until it returns True or raises an exception.
    while not context.join():
        time.sleep(0.1)
