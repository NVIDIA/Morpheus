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

import os

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


class DFEncoderDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        """A Custom DataLoader that unbatch the input data if batch_size is set to 1. """
        super().__init__(*args, **kwargs)

    def __iter__(self):
        """Iterate over the input data and unbatch it if batch_size is set to 1.

        Yields
        ------
        Dict[str, Union[int, Dict[str, torch.Tensor]]]
            A dictionary containing the unbatched input data of the current batch. 
            Example: {"batch_index": 0, "data": {"data1": tensor1, "data2": tensor2}}
        """
        for data_d in super().__iter__():
            if self.batch_size == 1:
                # unbatch to get rid of the first dimention of 1 intorduced by DataLoaders batching
                # (if batch size is set to 1)
                data_d["data"] = {k: v[0] if type(v) != list else [_v[0] for _v in v] for k, v in data_d["data"].items()}
            yield data_d

    @staticmethod
    def get_distributed_training_dataloader_from_dataset(dataset, rank, world_size, pin_memory=False, num_workers=0):
        """Returns a distributed training DataLoader given a dataset and other arguments.

        Parameters
        ----------
        dataset : Dataset
            The dataset to load the data from.
        rank : int
            The rank of the current process.
        world_size : int
            The number of processes to distribute the data across.
        pin_memory : bool, optional
            Whether to pin memory when loading data, by default False.
        num_workers : int, optional
            The number of worker processes to use for loading data, by default 0.

        Returns
        -------
        DataLoader
            The training DataLoader with DistributedSampler for distributed training.
        """
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        dataloader = DFEncoderDataLoader(
            dataset,
            batch_size=1,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=False,
            shuffle=False,
            sampler=sampler,
        )
        return dataloader

    @staticmethod
    def get_distributed_training_dataloader_from_path(model,
                                                      data_folder,
                                                      rank,
                                                      world_size,
                                                      load_data_fn=pd.read_csv,
                                                      pin_memory=False,
                                                      num_workers=0):
        """A helper funtion to get a distributed training DataLoader given a path to a folder containing data.

        Parameters
        ----------
        model : AutoEncoder
            The autoencoder model used to get relevant params and the preprocessing func.
        data_folder : str
            The path to the folder containing the data.
        rank : int
            The rank of the current process.
        world_size : int
            The number of processes to distribute the data across.
        load_data_fn : function, optional
            A function for loading data from a provided file path into a pandas.DataFrame, by default pd.read_csv.
        pin_memory : bool, optional
            Whether to pin memory when loading data, by default False.
        num_workers : int, optional
            The number of worker processes to use for loading data, by default 0.

        Returns
        -------
        DFEncoderDataLoader
            The training DataLoader with DistributedSampler for distributed training.
        """
        dataset = FileSystemDataset(
            data_folder,
            model.batch_size,
            model.preprocess_training_data,
            load_data_fn=load_data_fn,
        )
        dataloader = DFEncoderDataLoader.get_distributed_training_dataloader_from_dataset(
            dataset=dataset,
            rank=rank,
            world_size=world_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        return dataloader

    @staticmethod
    def get_distributed_training_dataloader_from_df(model, df, rank, world_size, pin_memory=False, num_workers=0):
        """A helper funtion to get a distributed training DataLoader given a pandas dataframe.
        
        Parameters
        ----------
        model : AutoEncoder
            The autoencoder model used to get relevant params and the preprocessing func.
        df : pandas.DataFrame
            The pandas dataframe containing the data.
        rank : int
            The rank of the current process.
        world_size : int
            The number of processes to distribute the data across.
        pin_memory : bool, optional
            Whether to pin memory when loading data, by default False.
        num_workers : int, optional
            The number of worker processes to use for loading data, by default 0.

        Returns
        -------
        DFEncoderDataLoader
            The training DataLoader with DistributedSampler for distributed training.
        """
        dataset = DataframeDataset.get_train_dataset(model, df)
        dataloader = DFEncoderDataLoader.get_distributed_training_dataloader_from_dataset(
            dataset=dataset,
            rank=rank,
            world_size=world_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        return dataloader


class FileSystemDataset(Dataset):
    """ A dataset class that reads data in batches from a folder and applies preprocessing to each batch.
    * This class assumes that the data is saved in small csv files in one folder.
    """

    def __init__(
        self,
        data_folder,
        batch_size=128,
        preprocess_fn=lambda x: x,
        load_data_fn=pd.read_csv,
        shuffle_rows_in_batch=True,
        shuffle_batch_indices=False,
        preload_data_into_memory=False,
    ):
        """Initialize a `DatasetFromPath` object.

        Parameters
        ----------
        data_folder : str
            The path to the folder containing the data files.
        batch_size : int
            The size of the batches to read the data in.
        preprocess_fn : function
            A function to preprocess the data, which should take a pandas.DataFrame and a boolean indicating 
            whether to shuffle rows in batch or not, and return a dictionary containing the preprocessed data
        load_data_fn : function, optional
            A function for loading data from a provided file path into a pandas.DataFrame, by default pd.read_csv.
        shuffle_rows_in_batch : bool, optional
            Whether to shuffle the rows within each batch, by default True.
        shuffle_batch_indices : bool, optional
            Whether to shuffle the order when iterating through the dataset (affects the __iter__ functionality)
            Should not matter when the dataset is fed into a dataloader and not used directly in the training loop.
        preload_data_into_memory : bool, optional
            Whether to preload all the data into memory, by default False.
            (Can speed up data loading if the data can fit into memory)
        """
        self._data_folder = data_folder
        self._filenames = sorted(os.listdir(data_folder))
        self._preprocess_fn = preprocess_fn
        self._load_data_fn = load_data_fn

        self._preloaded_data = None
        if preload_data_into_memory:
            self._preloaded_data = {fn: self._load_data_fn(f"{self._data_folder}/{fn}") for fn in self._filenames}

        self._file_sizes = {
            fn: self._get_file_len(fn) - 1 if not self._preloaded_data else len(self._preloaded_data[fn])
            for fn in self._filenames
        }
        self._count = sum(v for v in self._file_sizes.values())
        self._batch_size = batch_size
        self._shuffle_rows_in_batch = shuffle_rows_in_batch
        self._shuffle_batch_indices = shuffle_batch_indices

    def _get_file_len(self, fn, file_include_header_line=True):
        """Private method for getting the number of lines in a file.

        Parameters
        ----------
        fn : str
            The name of the file to get the length of.
        file_include_header_line : bool, optional
            Whether the file includes a header line, by default True.

        Returns
        -------
        int
            The number of lines in the file.
        """
        with open(f"{self._data_folder}/{fn}") as f:
            count = sum(1 for _ in f)
        return count - 1 if file_include_header_line else count

    @property
    def num_samples(self):
        """Returns the number of samples in the dataset. """
        return sum(self._file_sizes.values())

    def __len__(self):
        """Returns the number of batches in the dataset.
        Under normal circumstances, `Dataset` loads/returns one sample at a time. However, to optimize the loading of
        high-volume csv files, this class loads a batch of csv rows at a time. So this built-in len function needs to 
        return the batch count instead of sample count.

        Returns
        -------            
        int
            Number of batches in the dataset.
        """
        return int(np.ceil(self._count / self._batch_size))

    def __iter__(self):
        """Iterates through the whole dataset and yeild one batch at a time. The iteration order depends on 
        self.shuffle_batch_indices. Iterate in order if False, random order otherwise.

        Yields
        ------
        Dict[str, Union[int, Dict[str, torch.Tensor]]]
            A dictionary containing the preprocessed data for the current batch. 
            Example: {"batch_index": 0, "data": {"data1": tensor1, "data2": tensor2}}
        """
        indices = range(len(self))
        if self._shuffle_batch_indices:
            indices = np.arange(len(self))
            np.random.shuffle(indices)
        for i in indices:
            yield self[i]

    def __getitem__(self, idx):
        """Gets the item at the given index in the dataset.

        Parameters
        ----------
        idx : int
            The index of the item to get.

        Returns
        -------
        Dict[str, Union[int, Dict[str, torch.Tensor]]]
            A dictionary containing the preprocessed data for the current batch. 
            Example: {"batch_index": 0, "data": {"data1": tensor1, "data2": tensor2}}
        """
        start = idx * self._batch_size
        end = (idx + 1) * self._batch_size

        data = []
        curr_cnt = 0
        for fn in self._filenames:
            f_count = self._file_sizes[fn]
            curr_cnt = f_count

            if start < curr_cnt and end <= curr_cnt:
                data.append(self._get_data_from_filename(fn)[start:end])
                return self._preprocess(pd.concat(data), batch_index=idx)

            if start < curr_cnt and end > curr_cnt:
                data.append(self._get_data_from_filename(fn)[start:])
            start = max(0, start - curr_cnt)
            end = end - curr_cnt

        # clear out last batch
        return self._preprocess(pd.concat(data), batch_index=idx)

    def _get_data_from_filename(self, filename):
        """Returns the data from the given file as a pandas.DataFrame.

        Parameters
        ----------
        filename : str
            The filename of the file to load.

        Returns
        -------
        pandas.DataFrame
            Loaded data.
        """
        if self._preloaded_data:
            return self._preloaded_data[filename]
        return self._load_data_fn(f"{self._data_folder}/{filename}")

    def _preprocess(self, df, batch_index):
        """Preprocesses the given dataframe and returns a dictionary containing the preprocessed data.

        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe to preprocess.
        batch_index : int
            The index of the current batch.

        Returns
        -------
        Dict[str, Union[int, Dict[str, torch.Tensor]]]
            A dictionary containing the preprocessed data for the current batch. 
            Example: {"batch_index": 0, "data": {"data1": tensor1, "data2": tensor2}}
        """
        data = self._preprocess_fn(
            df,
            shuffle_rows_in_batch=self._shuffle_rows_in_batch,
        )
        return {"batch_index": batch_index, "data": data}

    def get_preloaded_data(self):
        """Loads all data from the files into memory and returns it as a pandas.DataFrame. """
        if self._preloaded_data is None:
            self._preloaded_data = {fn: self._load_data_fn(f"{self._data_folder}/{fn}") for fn in self._filenames}
        return pd.concat(pdf for pdf in self._preloaded_data.values())

    @property
    def preprocess_fn(self):
        return self._preprocess_fn

    @preprocess_fn.setter
    def preprocess_fn(self, value):
        self._preprocess_fn = value

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def shuffle_rows_in_batch(self):
        return self._shuffle_rows_in_batch

    @shuffle_rows_in_batch.setter
    def shuffle_rows_in_batch(self, value):
        self._shuffle_rows_in_batch = value

    @property
    def shuffle_batch_indices(self):
        return self._shuffle_batch_indices

    @shuffle_batch_indices.setter
    def shuffle_batch_indices(self, value):
        self._shuffle_batch_indices = value


class DataframeDataset(Dataset):

    def __init__(
        self,
        df,
        batch_size=128,
        preprocess_fn=lambda x: x,
        shuffle_rows_in_batch=True,
        shuffle_batch_indices=False,
    ):
        """A dataset class that slice a given dataframe into batches and applies preprocessing to each batch.
        * This class is developed to match the interface of the DatasetFromPath class. 
          As a result, unlike other common implementations of PyTorch datasets that return one row at a time and
          let the higher level DataLoader batch the data, this class returns one batch at a time when `__getitem__`
          is called. (Even if this limits the ability to fully shuffle the whole dataset.)

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe used for the dataset.
        batch_size : int
            The size of the batches to read the data in.
        preprocess_fn : function
            A function to preprocess the data, which should take a pandas.DataFrame and a boolean indicating whether 
            to shuffle rows in batch or not, and return a dictionary containing the preprocessed data.
        shuffle_rows_in_batch : bool, optional
            Whether to shuffle the rows within each batch, by default True.
        shuffle_batch_indices : bool, optional
            Whether to shuffle the order when iterating through the dataset (affects the __iter__ functionality)
            Should not matter when the dataset is fed into a dataloader and not used directly in the training loop.
        """
        self._df = df
        self._preprocess_fn = preprocess_fn

        self._count = len(self._df)
        self._batch_size = batch_size
        self._shuffle_rows_in_batch = shuffle_rows_in_batch
        self._shuffle_batch_indices = shuffle_batch_indices

    @property
    def num_samples(self):
        """Returns the number of samples in the dataset. """
        return len(self._df)

    def __len__(self):
        """Returns the number of batches in the dataset.
        Under normal circumstances, `Dataset` loads/returns one sample at a time. However, to match the behavior of the
        DatasetFromPath class, this class returns a batch of data when queried. So this built-in len function needs to 
        return the batch count instead of sample count.

        Returns
        -------
        int
            Number of batches in the dataset.
        """
        return int(np.ceil(self._count / self._batch_size))

    def __iter__(self):
        """Iterates through the whole dataset and yeild one batch at a time. The iteration order depends on 
        self.shuffle_batch_indices. Iterate in order if False, random order otherwise.

        Yields
        ------
        Dict[str, Union[int, Dict[str, torch.Tensor]]]
            A dictionary containing the preprocessed data for the current batch. 
            Example: {"batch_index": 0, "data": {"data1": tensor1, "data2": tensor2}}
        """
        indices = range(len(self))
        if self._shuffle_batch_indices:
            indices = np.arange(len(self))
            np.random.shuffle(indices)

        for i in indices:
            yield self[i]

    def __getitem__(self, idx):
        """Gets the item (batch) at the given index in the dataset.

        Parameters
        ----------
        idx : int
            The index of the item to get.

        Returns
        -------
        Dict[str, Union[int, Dict[str, torch.Tensor]]]
            A dictionary containing the preprocessed data for the current batch. 
            Example: {"batch_index": 0, "data": {"data1": tensor1, "data2": tensor2}}
        """
        start = idx * self._batch_size
        end = (idx + 1) * self._batch_size

        data = self._df[start:end]
        return self._preprocess(data, batch_index=idx)

    def _preprocess(self, df, batch_index):
        """Preprocesses the given dataframe and returns a dictionary containing the preprocessed data.

        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe to preprocess.
        batch_index : int
            The index of the current batch.

        Returns
        -------
        Dict[str, Union[int, Dict[str, torch.Tensor]]]
            A dictionary containing the preprocessed data for the current batch. 
            Example: {"batch_index": 0, "data": {"data1": tensor1, "data2": tensor2}}
        """
        data = self._preprocess_fn(
            df,
            shuffle_rows_in_batch=self._shuffle_rows_in_batch,
        )
        return {"batch_index": batch_index, "data": data}

    @property
    def preprocess_fn(self):
        return self._preprocess_fn

    @preprocess_fn.setter
    def preprocess_fn(self, value):
        self._preprocess_fn = value

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def shuffle_rows_in_batch(self):
        return self._shuffle_rows_in_batch

    @shuffle_rows_in_batch.setter
    def shuffle_rows_in_batch(self, value):
        self._shuffle_rows_in_batch = value

    @property
    def shuffle_batch_indices(self):
        return self._shuffle_batch_indices

    @shuffle_batch_indices.setter
    def shuffle_batch_indices(self, value):
        self._shuffle_batch_indices = value
