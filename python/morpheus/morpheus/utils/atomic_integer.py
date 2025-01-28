# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

import threading


class AtomicInteger():
    """
    Simple atomic integer from https://stackoverflow.com/a/48433648/634820

    Parameters
    ----------
    _value : int
        Initial value, defaults to 0.
    """

    def __init__(self, value=0):
        self._value = int(value)
        self._lock = threading.Lock()

    def inc(self, inc_val=1):
        """
        Increments and returns new value.

        Parameters
        ----------
        inc_val : int, optional
            Value to increment by, default 1.

        Returns
        -------
        int
            Incremented value.
        """
        with self._lock:
            self._value += int(inc_val)
            return self._value

    def dec(self, dec_val=1):
        """
        Decrements and returns new value.

        Parameters
        ----------
        dec_val : int, optional
            Value to decrement by, default 1.

        Returns
        -------
        int
            Decremented value.
        """
        return self.inc(-dec_val)

    def get_and_inc(self, inc_val=1):
        """
        Gets the current value, returns it, and increments. Different from `inc()` which increments, then returns.

        Parameters
        ----------
        inc_val : int, optional
            How much to increment, by default 1.

        Returns
        -------
        int
            Incremented value.
        """
        with self._lock:
            tmp_val = self._value
            self._value += int(inc_val)
            return tmp_val

    @property
    def value(self):
        """
        Get value.

        Returns
        -------
        int
            Current value.
        """
        with self._lock:
            return self._value

    @value.setter
    def value(self, val):
        """
        Set value.

        Parameters
        ----------
        val : int
            Set to this value.

        Returns
        -------
        int
            New value.
        """
        with self._lock:
            self._value = int(val)
            return self._value
