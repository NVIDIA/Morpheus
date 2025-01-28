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
"""File source stage."""

import logging
import pathlib
import typing

import mrc

from morpheus.cli import register_stage
from morpheus.common import FileTypes
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stage_schema import StageSchema

logger = logging.getLogger(__name__)


@register_stage("from-file", modes=[PipelineModes.FIL, PipelineModes.NLP, PipelineModes.OTHER])
class FileSourceStage(GpuAndCpuMixin, PreallocatorMixin, SingleOutputSource):
    """
    Load messages from a file.

    Source stage is used to load messages from a file and dumping the contents into the pipeline immediately. Useful for
    testing performance and accuracy of a pipeline.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    filename : pathlib.Path, exists = True, dir_okay = False
        Name of the file from which the messages will be read.
    iterative : boolean, default = False, is_flag = True
        Iterative mode will emit dataframes one at a time. Otherwise a list of dataframes is emitted. Iterative mode is
        good for interleaving source stages.
    file_type : `morpheus.common.FileTypes`, optional, case_sensitive = False
        Indicates what type of file to read. Specifying 'auto' will determine the file type from the extension.
        Supported extensions: 'csv', 'json', 'jsonlines' and 'parquet'.
    repeat : int, default = 1, min = 1
        Repeats the input dataset multiple times. Useful to extend small datasets for debugging.
    filter_null : bool, default = True
        Whether to filter rows with null `filter_null_columns` columns. Null values in source data  can cause issues
        down the line with processing. Setting this to True is recommended.
    filter_null_columns : list[str], default = None
        Column or columns to filter null values from. Ignored when `filter_null` is False. If None, and `filter_null`
        is `True`, this will default to `["data"]`
    parser_kwargs : dict, default = {}
        Extra options to pass to the file parser.
    """

    def __init__(self,
                 c: Config,
                 filename: pathlib.Path,
                 iterative: bool = False,
                 file_type: FileTypes = FileTypes.Auto,
                 repeat: int = 1,
                 filter_null: bool = True,
                 filter_null_columns: list[str] = None,
                 parser_kwargs: dict = None):

        super().__init__(c)

        self._batch_size = c.pipeline_batch_size

        self._filename = filename
        self._file_type = file_type
        self._filter_null = filter_null

        if filter_null_columns is None or len(filter_null_columns) == 0:
            filter_null_columns = ["data"]

        self._filter_null_columns = filter_null_columns

        self._parser_kwargs = parser_kwargs or {}

        self._input_count = None
        self._max_concurrent = c.num_threads

        # Iterative mode will emit dataframes one at a time. Otherwise a list of dataframes is emitted. Iterative mode
        # is good for interleaving source stages.
        self._iterative = iterative
        self._repeat_count = repeat

    @property
    def name(self) -> str:
        """Return the name of the stage"""
        return "from-file"

    @property
    def input_count(self) -> int:
        """Return None for no max intput count"""
        return self._input_count

    def supports_cpp_node(self) -> bool:
        """Indicates whether this stage supports a C++ node"""
        return True

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(MessageMeta)

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:

        if self._build_cpp_node():
            import morpheus._lib.stages as _stages
            node = _stages.FileSourceStage(builder,
                                           self.unique_name,
                                           self._filename,
                                           self._repeat_count,
                                           self._filter_null,
                                           self._filter_null_columns,
                                           self._parser_kwargs)
        else:
            node = builder.make_source(self.unique_name, self._generate_frames)

        return node

    def _generate_frames(self, subscription: mrc.Subscription) -> typing.Iterable[MessageMeta]:

        df = read_file_to_df(
            self._filename,
            self._file_type,
            filter_nulls=self._filter_null,
            filter_null_columns=self._filter_null_columns,
            parser_kwargs=self._parser_kwargs,
            df_type=self.df_type_str,
        )

        for i in range(self._repeat_count):
            if not subscription.is_subscribed():
                break

            x = MessageMeta(df)

            # If we are looping, copy the object. Do this before we push the object in case it changes
            if (i + 1 < self._repeat_count):
                df = df.copy()

                # Shift the index to allow for unique indices without reading more data
                df.index += len(df)

            yield x
