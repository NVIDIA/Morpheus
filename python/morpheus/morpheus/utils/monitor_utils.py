# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

import logging

from tqdm import TMonitor
from tqdm import TqdmSynchronisationWarning
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MorpheusTqdmMonitor(TMonitor):
    """
    Monitoring thread for tqdm bars.
    """

    def run(self):
        """
        This function does exactly the same as `TMonitor.run`, except we do not check for `instance.miniters == 1`
        before updating. This allows the timer to update every 1 second on the screen making the pipeline feel running
        """
        cur_t = self._time()
        while True:
            # After processing and before sleeping, notify that we woke
            # Need to be done just before sleeping
            self.woken = cur_t
            # Sleep some time...
            self.was_killed.wait(self.sleep_interval)
            # Quit if killed
            if self.was_killed.is_set():
                return
            # Then monitor!
            # Acquire lock (to access _instances)
            with self.tqdm_cls.get_lock():
                cur_t = self._time()
                # Check tqdm instances are waiting too long to print
                instances = self.get_instances()
                for instance in instances:
                    # Check event in loop to reduce blocking time on exit
                    if self.was_killed.is_set():
                        return
                    # Only if mininterval > 1 (else iterations are just slow)
                    # and last refresh exceeded maxinterval
                    if ((cur_t - instance.last_print_t) >= instance.maxinterval):
                        # Refresh now! (works only for manual tqdm)
                        instance.refresh(nolock=True)
                    # Remove accidental long-lived strong reference
                    del instance
                if instances != self.get_instances():  # pragma: nocover
                    logging.warning("Set changed size during iteration" +
                                    " (see https://github.com/tqdm/tqdm/issues/481)",
                                    TqdmSynchronisationWarning,
                                    stacklevel=2)
                # Remove accidental long-lived strong references
                del instances


class MorpheusTqdm(tqdm):
    """
    Subclass of tqdm to provide slightly different functionality with their TMonitor.

    """
    monitor_interval = 1  # set to 0 to disable the thread
    monitor: MorpheusTqdmMonitor = None

    def __new__(cls, *args, **kwargs):
        with cls.get_lock():  # also constructs lock if non-existent

            if (cls.monitor is None or not cls.monitor.report()):
                # Set the new type of monitor
                cls.monitor = MorpheusTqdmMonitor(cls, cls.monitor_interval)

        return tqdm.__new__(cls, args, kwargs)

    def __init__(self, *args, **kwargs):

        # Must set this first
        self.is_running = True

        super().__init__(*args, **kwargs)

        self.last_update_t = self.start_t

    @property
    def format_dict(self):

        base_val = super().format_dict

        # If we arent running, dont increment the time
        if (not self.is_running):
            base_val["elapsed"] = self.last_update_t - self.start_t

        return base_val

    def update(self, n=1):
        """
        This function updates the time and progress bar.

        Parameters
        ----------
        n : int
            Increment to add to the internal counter of iterations.
        """

        self.last_update_t = self._time()

        return super().update(n)

    def stop(self):
        """
        Progress bar incrementing is stopped by this function.
        """
        # Set is running to false to stop elapsed from incrementing
        self.is_running = False


class SilentMorpheusTqdm(MorpheusTqdm):
    """
    Subclass of MorpheusTqdm to silent monitors, it provides slightly different functionality with their TMonitor.

    """

    def refresh(self, nolock=False, lock_args=None):
        return
