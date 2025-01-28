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

import logging

log = logging.getLogger(__name__)


class DataLoader():
    """Wrapper class is used to return dataframe partitions based on batchsize."""

    def __init__(self, dataset, batchsize=1000):
        """Constructor to create dataframe partitions.

        :param df: Input dataframe.
        :type df: cudf.DataFrame
        :param batch_size: Number of records in the dataframe.
        :type batch_size: int
        """
        self.__dataset = dataset
        self.__batchsize = batchsize

    @property
    def dataset_len(self):
        return self.__dataset.length

    @property
    def dataset(self):
        return self.__dataset

    def get_chunks(self):
        """ A generator function that yields each chunk of original input dataframe based on batchsize
        :return: Partitioned dataframe.
        :rtype: cudf.DataFrame
        """
        prev_chunk_offset = 0
        while prev_chunk_offset < self.__dataset.length:
            curr_chunk_offset = prev_chunk_offset + self.__batchsize
            chunk = self.__dataset.data[prev_chunk_offset:curr_chunk_offset:1]
            prev_chunk_offset = curr_chunk_offset
            yield chunk
