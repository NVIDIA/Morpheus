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

import os
from abc import abstractmethod
from functools import partial

import mrc
import pandas as pd
from mrc.core import operators as ops

from morpheus.common import FileTypes
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.utils.directory_watcher import DirectoryWatcher


class AutoencoderSourceStage(PreallocatorMixin, GpuAndCpuMixin, SingleOutputSource):
    """
    All AutoEncoder source stages must extend this class and implement the `files_to_dfs_per_user` abstract method.
    Feature columns can be managed by overriding the `derive_features` method. Otherwise, all columns from input
    data pass through to next stage.

    Extend this class to load messages from a files and dump contents into a DFP pipeline immediately. Useful for
    testing performance and accuracy of a pipeline.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    input_glob : str
        Input glob pattern to match files to read. For example, `./input_dir/*.json` would read all files with the
        'json' extension in the directory input_dir.
    watch_directory : bool, default = False
        The watch directory option instructs this stage to not close down once all files have been read. Instead it will
        read all files that match the 'input_glob' pattern, and then continue to watch the directory for additional
        files. Any new files that are added that match the glob will then be processed.
    max_files: int, default = -1
        Max number of files to read. Useful for debugging to limit startup time. Default value of -1 is unlimited.
    file_type : `morpheus.common.FileTypes`, default = 'FileTypes.Auto'.
        Indicates what type of file to read. Specifying 'auto' will determine the file type from the extension.
        Supported extensions: 'json', 'csv'
    repeat: int, default = 1
        How many times to repeat the dataset. Useful for extending small datasets in debugging.
    sort_glob : bool, default = False
        If true the list of files matching `input_glob` will be processed in sorted order.
    recursive: bool, default = True
        If true, events will be emitted for the files in subdirectories that match `input_glob`.
    queue_max_size: int, default = 128
        Maximum queue size to hold the file paths to be processed that match `input_glob`.
    batch_timeout: float, default = 5.0
        Timeout to retrieve batch messages from the queue.
    """

    def __init__(self,
                 c: Config,
                 input_glob: str,
                 watch_directory: bool = False,
                 max_files: int = -1,
                 file_type: FileTypes = FileTypes.Auto,
                 repeat: int = 1,
                 sort_glob: bool = False,
                 recursive: bool = True,
                 queue_max_size: int = 128,
                 batch_timeout: float = 5.0):

        SingleOutputSource.__init__(self, c)

        self._input_glob = input_glob
        self._file_type = file_type

        self._feature_columns = c.ae.feature_columns
        self._user_column_name = c.ae.userid_column_name
        self._userid_filter = c.ae.userid_filter

        self._input_count = None

        # Hold the max index we have seen to ensure sequential and increasing indexes
        self._rows_per_user: dict[str, int] = {}

        # Iterative mode will emit dataframes one at a time. Otherwise a list of dataframes is emitted. Iterative mode
        # is good for interleaving source stages.
        self._repeat_count = repeat

        self._df_class = self.get_df_class()

        self._watcher = DirectoryWatcher(input_glob=input_glob,
                                         watch_directory=watch_directory,
                                         max_files=max_files,
                                         sort_glob=sort_glob,
                                         recursive=recursive,
                                         queue_max_size=queue_max_size,
                                         batch_timeout=batch_timeout,
                                         should_stop_fn=self.is_stop_requested)

    @property
    def input_count(self) -> int:
        """Return None for no max input count"""
        return self._input_count if self._input_count is not None else 0

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(ControlMessage)

    def get_match_pattern(self, glob_split):
        """Return a file match pattern"""
        dir_to_watch = os.path.dirname(glob_split[0])
        match_pattern = self._input_glob.replace(dir_to_watch + "/", "", 1)

        return match_pattern

    @staticmethod
    def repeat_df(df: pd.DataFrame, repeat_count: int) -> list[pd.DataFrame]:
        """
        This function iterates over the same dataframe to extending small datasets in debugging with incremental
        updates to the `event_dt` and `eventTime` columns.

        Parameters
        ----------
        df : pd.DataFrame
            To be repeated dataframe.
        repeat_count : int
            Number of times the given dataframe should be repeated.

        Returns
        -------
        df_array : list[pd.DataFrame]
            List of repeated dataframes.
        """

        df_array = []

        df_array.append(df)

        for _ in range(1, repeat_count):
            x = df.copy()

            # Now increment the timestamps by the interval in the df
            x["event_dt"] = x["event_dt"] + (x["event_dt"].iloc[-1] - x["event_dt"].iloc[0])
            x["eventTime"] = x["event_dt"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

            df_array.append(x)

            # Set df for next iteration
            df = x

        return df_array

    @staticmethod
    def batch_user_split(x: list[pd.DataFrame],
                         userid_column_name: str,
                         userid_filter: str,
                         datetime_column_name="event_dt"):
        """
        Creates a dataframe for each userid.

        Parameters
        ----------
        x : list[pd.DataFrame]
            List of dataframes.
        userid_column_name : str
            Name of a dataframe column used for categorization.
        userid_filter : str
            Only rows with the supplied userid are filtered.
        datetime_column_name : str
            Name of the dataframe column used to sort the rows.

        Returns
        -------
        user_dfs : dict[str, pd.DataFrame]
            Dataframes, each of which is associated with a single userid.
        """

        combined_df = pd.concat(x)

        if (datetime_column_name in combined_df):

            # Convert to date_time column
            # combined_df["event_dt"] = pd.to_datetime(combined_df["eventTime"])

            # Set the index name so we can sort first by time then by index (to keep things all in order). Then restore
            # the name
            saved_index_name = combined_df.index.name

            combined_df.index.name = "idx"

            # Sort by time
            combined_df = combined_df.sort_values(by=[datetime_column_name, "idx"])

            combined_df.index.name = saved_index_name

        # Get the users in this DF
        unique_users = combined_df[userid_column_name].unique()

        user_dfs = {}

        for user_name in unique_users:

            if (userid_filter is not None and user_name != userid_filter):
                continue

            # Get just this users data and make a copy to remove link to grouped DF
            user_df = combined_df[combined_df[userid_column_name] == user_name].copy()

            user_dfs[user_name] = user_df

        return user_dfs

    @staticmethod
    @abstractmethod
    def files_to_dfs_per_user(x: list[str],
                              userid_column_name: str,
                              feature_columns: list[str],
                              userid_filter: str = None,
                              repeat_count: int = 1) -> dict[str, pd.DataFrame]:
        """
        Stages that extend `AutoencoderSourceStage` must implement this abstract function
        in order to convert messages in the files to dataframes per userid.

        Parameters
        ----------
        x : list[str]
            List of messages.
        userid_column_name : str
            Name of the column used for categorization.
        feature_columns : list[str]
            Feature column names.
        userid_filter : str
            Only rows with the supplied userid are filtered.
        repeat_count : str
            Number of times the given rows should be repeated.

        Returns
        -------
            : dict[str, pd.DataFrame]
            Dataframe per userid.
        """

        pass

    @staticmethod
    def derive_features(df: pd.DataFrame, feature_columns: list[str] | None):  # pylint: disable=unused-argument
        """
        If any features are available to be derived, can be implemented by overriding this function.

        Parameters
        ----------
        df : pd.DataFrame
            A dataframe.
        feature_columns : list[str]
            Names of columns that are need to be derived.

        Returns
        -------
        df : list[pd.DataFrame]
            Dataframe with actual and derived columns.
        """
        return df

    def _add_derived_features(self, user_dataframes: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:

        for user_name in user_dataframes.keys():
            user_dataframes[user_name] = self.derive_features(user_dataframes[user_name], None)

        return user_dataframes

    def _build_message(self, user_dataframes: dict[str, pd.DataFrame]) -> list[ControlMessage]:

        messages = []

        for user_name, user_df in user_dataframes.items():

            # See if we have seen this user before
            if (user_name not in self._rows_per_user):
                self._rows_per_user[user_name] = 0

            # Combine the original index with itself so it shows up as a named column
            user_df.index.name = "_index_" + (user_df.index.name or "")
            user_df = user_df.reset_index()

            # Now ensure the index for this user is correct
            user_df.index = range(self._rows_per_user[user_name], self._rows_per_user[user_name] + len(user_df))
            self._rows_per_user[user_name] += len(user_df)

            # If we're in GPU mode we need to convert to cuDF
            if not isinstance(user_df, self._df_class):
                for col in [col for col in user_df.columns if isinstance(user_df[col].dtype, pd.DatetimeTZDtype)]:
                    user_df[col] = user_df[col].dt.tz_convert(None)

                user_df = self._df_class(user_df)

            # Now make a message with the user name in metadata
            meta = MessageMeta(user_df)
            message = ControlMessage()
            message.payload(meta)
            message.set_metadata("user_id", user_name)

            messages.append(message)

        return messages

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:
        # The first source just produces filenames
        return self._watcher.build_node(self.unique_name, builder)

    def _post_build_single(self, builder: mrc.Builder, out_node: mrc.SegmentObject) -> mrc.SegmentObject:

        # At this point, we have batches of filenames to process. Make a node for processing batches of
        # filenames into batches of dataframes
        post_node = builder.make_node(
            self.unique_name + "-post",
            ops.map(
                partial(
                    self.files_to_dfs_per_user,
                    userid_column_name=self._user_column_name,
                    feature_columns=None,  # Use None here to leave all columns in
                    userid_filter=self._userid_filter,
                    repeat_count=self._repeat_count)),
            ops.map(self._add_derived_features),
            # Now group the batch of dataframes into a single df, split by user, and send a single ControlMessage
            # per user
            ops.map(self._build_message),
            ops.flatten())
        builder.make_edge(out_node, post_node)

        return super()._post_build_single(builder, post_node)
