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
"""Utilities for modifying the environment for testing Morpheus"""

import contextlib
import os


@contextlib.contextmanager
def set_env(**env_vars):
    """
    Temporarily updates the ``os.environ`` dictionary in-place.

    The ``os.environ`` dictionary is updated in-place so that the modification
    is sure to work in all situations.

    Setting a value to ``None`` will cause the key to be removed from the environment.
    """
    # Taken from https://stackoverflow.com/a/34333710
    env = os.environ

    # Remove any which are set to None
    remove = [k for k, v in env_vars.items() if v is None]

    # Save the remaining environment variables to set
    update = {k: v for k, v in env_vars.items() if v is not None}

    # List of environment variables being updated or removed.
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # Environment variables and values to restore on exit.
    update_after = {k: env[k] for k in stomped}
    # Environment variables and values to remove on exit.
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)

        for k in remove:
            env.pop(k, None)

        yield
    finally:
        env.update(update_after)

        for k in remove_after:
            env.pop(k)
