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
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class DataManagerDataset(Dataset):
    """Custom Dataset for loading data from a DataManager instance."""

    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.ids = list(data_manager.records.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load data from DataManager
        data = self.data_manager.load(self.ids[idx])

        # Convert to PyTorch tensor
        # Here it's assumed that data is a DataFrame with a single numerical column.
        # Modify as needed to match your actual data format.
        data = torch.tensor(data.values)

        return data


class DataManagerLoader:
    """Wrapper around DataManager to produce a PyTorch DataLoader."""

    def __init__(self, data_manager, batch_size=1, shuffle=False, num_workers=0):
        self.dataset = DataManagerDataset(data_manager)
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def get_data_loader(self):
        return self.data_loader
