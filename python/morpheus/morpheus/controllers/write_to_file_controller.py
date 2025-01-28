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

import os

import mrc
import mrc.core.operators as ops

from morpheus.common import FileTypes
from morpheus.common import determine_file_type
from morpheus.io import serializers
from morpheus.messages import MessageMeta
from morpheus.utils.type_aliases import DataFrameType


class WriteToFileController:
    """
    Controller class for writing data to a file with customizable options.

    Parameters
    ----------
    filename : str
        The output file name.
    overwrite : bool
        Flag to indicate whether to overwrite an existing file.
    file_type : FileTypes
        The type of the output file (e.g., CSV, JSON).
    include_index_col : bool
        Flag to indicate whether to include the index column in the output.
    flush : bool
        Flag to indicate whether to flush the output file after writing.
    """

    def __init__(self, filename: str, overwrite: bool, file_type: FileTypes, include_index_col: bool, flush: bool):
        self._output_file = filename
        self._overwrite = overwrite

        if (os.path.exists(self._output_file)):
            if (self._overwrite):
                os.remove(self._output_file)
            else:
                raise FileExistsError(
                    f"Cannot output classifications to '{self._output_file}'. File exists and overwrite = False")
        else:
            # Ensure our directory exists
            os.makedirs(os.path.realpath(os.path.dirname(self._output_file)), exist_ok=True)

        self._file_type = file_type

        if (self._file_type == FileTypes.Auto):
            self._file_type = determine_file_type(self._output_file)

        self._is_first = True
        self._include_index_col = include_index_col
        self._flush = flush

    @property
    def output_file(self):
        """
        Get the output file name.
        """
        return self._output_file

    @property
    def overwrite(self):
        """
        Get the flag indicating whether to overwrite an existing file.
        """
        return self._overwrite

    @property
    def file_type(self):
        """
        Get the type of the output file.
        """
        return self._file_type

    @property
    def include_index_col(self):
        """
        Get the flag indicating whether to include the index column in the output.
        """
        return self._include_index_col

    @property
    def flush(self):
        """
        Get the flag indicating whether to flush the output file after writing.
        """
        return self._flush

    def _convert_to_strings(self, df: DataFrameType):
        if self._file_type in (FileTypes.JSON, 'JSON'):
            output_strs = serializers.df_to_json(df, include_index_col=self._include_index_col)
        elif self._file_type in (FileTypes.CSV, 'CSV'):
            output_strs = serializers.df_to_csv(df,
                                                include_header=self._is_first,
                                                include_index_col=self._include_index_col)
            self._is_first = False
        else:
            raise NotImplementedError(f"Unknown file type: {self._file_type}")

        # Remove any trailing whitespace
        if (len(output_strs[-1].strip()) == 0):
            output_strs = output_strs[:-1]

        return output_strs

    def node_fn(self, obs: mrc.Observable, sub: mrc.Subscriber):

        # When writing to a parquet file, we need to open the file in binary mode
        if self._file_type == FileTypes.PARQUET:
            with open(self._output_file, "wb") as out_file:

                def write_to_file(x: MessageMeta):

                    x.df.to_parquet(out_file)

                    if self._flush:
                        out_file.flush()

                    return x

            obs.pipe(ops.map(write_to_file)).subscribe(sub)

        else:
            with open(self._output_file, "a", encoding='UTF-8') as out_file:

                def write_to_file(x: MessageMeta):

                    lines = self._convert_to_strings(x.df)
                    out_file.writelines(lines)

                    if self._flush:
                        out_file.flush()

                    return x

                obs.pipe(ops.map(write_to_file)).subscribe(sub)
