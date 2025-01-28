# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch


class DistributedAutoEncoder(torch.nn.parallel.DistributedDataParallel):

    def __init__(self, *args, **kwargs):
        self.__dict__['initialized'] = False

        # set placeholder attribute for the pytorch module, which will be
        # populated in the init function of the super class
        self.module = None
        super().__init__(*args, **kwargs)

        self.__dict__['initialized'] = True

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def __setattr__(self, name, value):
        # set the attribute to the instance it belongs to after init
        if not self.initialized or hasattr(self, name):
            super().__setattr__(name, value)
        else:
            setattr(self.module, name, value)
