# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import cupy as cp
import numpy as np
import torch


def manual_seed(seed: int, cpu_only: bool = False):
    """
    Manually see the random number generators for the Python standard lib, PyTorch, NumPy and CuPy

    Parameters
    ----------
    seed : int
        The seed value to use
    cpu_only : bool, default = False
        When set to True, CuPy and CUDA specific PyTorch settings are not set.
    """
    random.seed(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)

    if not cpu_only:
        cp.random.seed(seed)

        torch.cuda.manual_seed_all(seed)  # the "all" refers to all GPUs

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
