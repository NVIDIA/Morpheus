# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
import time
import typing
from functools import partial
from urllib.parse import urlsplit

import fsspec
import mrc
import s3fs
from mrc.core import operators as ops

from morpheus.cli import register_stage
from morpheus.common import FileTypes
from morpheus.config import Config
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages import MessageMeta
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


@register_stage("file-source")
class FileSource(PreallocatorMixin, SingleOutputSource):
    """
    Load messages from a file.

    FileSource is used to produce messages loaded from a file. Useful for testing performance and
    accuracy of a pipeline.

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    files : List[str]
        List of paths to be read from, can be a list of S3 URLs (`s3://path`) and can include wildcard characters `*`
        as defined by `fsspec`:
        https://filesystem-spec.readthedocs.io/en/latest/api.html?highlight=open_files#fsspec.open_files
    watch : bool, default = False
        When True, will check `files` for new files and emit them as they appear.
    watch_interval : float, default = 1.0
        When `watch` is True, this is the time in seconds between polling the paths in `files` for new files.
    sort : bool, default = False
        When True, the list of files will be processed in sorted order.
    file_type : morpheus.common.FileTypes, optional, case_sensitive = False
        Indicates what type of file to read. Specifying 'auto' will determine the file type from the extension.
        Supported extensions: 'csv', 'json', 'jsonlines' and 'parquet'.
    parser_kwargs : dict, default = None
        Extra options to pass to the file parser.
    max_files : int, default = -1
        Max number of files to read. Useful for debugging to limit startup time. Default value of -1 is unlimited.
    storage_connection_kwargs : dict, default = None
        Extra settings that are relevant to a specific storage connection used by `fsspec.open_files`.
    """

    def __init__(self,
                 config: Config,
                 files: typing.List[str],
                 watch: bool = False,
                 watch_interval: float = 1.0,
                 sort: bool = False,
                 file_type: FileTypes = FileTypes.Auto,
                 parser_kwargs: dict = None,
                 max_files: int = -1,
                 storage_connection_kwargs: dict = None):

        super().__init__(config)

        if not files or len(files) == 0:
            raise ValueError("The 'files' cannot be empty.")

        if watch and len(files) != 1:
            raise ValueError("When 'watch' is True, the 'files' should contain exactly one file path.")

        self._files = list(files)
        self._protocols = self._extract_unique_protocols()

        if len(self._protocols) > 1:
            raise ValueError("Accepts same protocol input files, but it received multiple protocols.")

        self._watch = watch
        self._sort = sort
        self._file_type = file_type
        self._parser_kwargs = parser_kwargs or {}
        self._storage_connection_kwargs = storage_connection_kwargs or {}
        self._watch_interval = watch_interval
        self._max_files = max_files

    @property
    def name(self) -> str:
        """Return the name of the stage."""
        return "file-source"

    def supports_cpp_node(self) -> bool:
        """Indicates whether or not this stage supports a C++ node."""
        return False

    def _extract_unique_protocols(self) -> set:
        """Extracts unique protocols from the given file paths."""
        protocols = set()

        for file in self._files:
            scheme = urlsplit(file).scheme
            if scheme:
                protocols.add(scheme.lower())
            else:
                protocols.add("file")

        return protocols

    def _build_source(self, builder: mrc.Builder) -> StreamPair:

        if self._build_cpp_node():
            raise RuntimeError("Does not support C++ nodes.")

        if self._watch:
            generator_function = self._polling_generate_frames_fsspec
        else:
            generator_function = self._generate_frames_fsspec

        out_stream = builder.make_source(self.unique_name, generator_function())
        out_type = fsspec.core.OpenFiles

        # Supposed to just return a source here
        return out_stream, out_type

    def _generate_frames_fsspec(self) -> typing.Iterable[fsspec.core.OpenFiles]:

        files: fsspec.core.OpenFiles = fsspec.open_files(self._files, **self._storage_connection_kwargs)

        if (len(files) == 0):
            raise RuntimeError(f"No files matched input strings: '{self._files}'. "
                               "Check your input pattern and ensure any credentials are correct.")

        if self._sort:
            files = sorted(files, key=lambda f: f.full_name)

        if self._max_files > 0:
            files = files[:self._max_files]

        yield files

    def _polling_generate_frames_fsspec(self) -> typing.Iterable[fsspec.core.OpenFiles]:
        files_seen = set()
        curr_time = time.monotonic()
        next_update_epoch = curr_time
        processed_files_count = 0
        has_s3_protocol = "s3" in self._protocols

        while (True):
            # Before doing any work, find the next update epoch after the current time
            while (next_update_epoch <= curr_time):
                # Only ever add `self._watch_interval` to next_update_epoch so all updates are at repeating intervals
                next_update_epoch += self._watch_interval

            file_set = set()
            filtered_files = []

            # Clear cached instance, otherwise we don't receive newly touched files.
            if has_s3_protocol:
                s3fs.S3FileSystem.clear_instance_cache()

            files = fsspec.open_files(self._files, **self._storage_connection_kwargs)

            for file in files:
                file_set.add(file.full_name)
                if file.full_name not in files_seen:
                    filtered_files.append(file)

            # Replace files_seen with the new set of files. This prevents a memory leak that could occurr if files are
            # deleted from the input directory. In addition if a file with a given name was created, seen/processed by
            # the stage, and then deleted, and a new file with the same name appeared sometime later, the stage will
            # need to re-ingest that new file.
            files_seen = file_set

            if len(filtered_files) > 0:

                if self._sort:
                    filtered_files = sorted(filtered_files, key=lambda f: f.full_name)

                if self._max_files > 0:
                    filtered_files = filtered_files[:self._max_files - processed_files_count]
                    processed_files_count += len(filtered_files)

                    if self._max_files <= processed_files_count:
                        logger.debug("Maximum file limit reached. Exiting polling service...")
                        yield fsspec.core.OpenFiles(filtered_files, fs=files.fs)
                        break

                yield fsspec.core.OpenFiles(filtered_files, fs=files.fs)

            curr_time = time.monotonic()

            # If we spent more than `self._watch_interval` doing work and/or yielding to the output channel blocked,
            # then we should only sleep for the remaining time until the next update epoch.
            sleep_duration = next_update_epoch - curr_time
            if (sleep_duration > 0):
                time.sleep(sleep_duration)
                curr_time = time.monotonic()

    @staticmethod
    def generate_frames(file: fsspec.core.OpenFile, file_type: FileTypes, parser_kwargs: dict) -> MessageMeta:
        """
        Generate message frame from a file.

        This function reads data from a file and generates message frames (MessageMeta) based on the file's content.
        It can be used to load and process messages from a file for testing and analysis within a Morpheus pipeline.

        Parameters
        ----------
        file : fsspec.core.OpenFile
            An open file object using fsspec.
        file_type : FileTypes
            Indicates the type of the file to read. Supported types include 'csv', 'json', 'jsonlines', and 'parquet'.
        parser_kwargs : dict
            Additional keyword arguments to pass to the file parser.

        Returns
        -------
        MessageMeta
            MessageMeta object, each containing a dataframe of messages from the file.
        """
        df = read_file_to_df(
            file.full_name,
            file_type=file_type,
            filter_nulls=False,
            parser_kwargs=parser_kwargs,
            df_type="cudf",
        )

        meta = MessageMeta(df)

        return meta

    def _post_build_single(self, builder: mrc.Builder, out_pair: StreamPair) -> StreamPair:

        out_stream = out_pair[0]

        post_node = builder.make_node(
            self.unique_name + "-post",
            ops.flatten(),  # Flatten list of open fsspec files
            ops.map(partial(self.generate_frames, file_type=self._file_type,
                            parser_kwargs=self._parser_kwargs))  # Generate dataframe for each file
        )

        builder.make_edge(out_stream, post_node)

        out_stream = post_node
        out_type = MessageMeta

        return super()._post_build_single(builder, (out_stream, out_type))
