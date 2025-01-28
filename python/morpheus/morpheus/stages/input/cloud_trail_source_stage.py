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

import logging
import os
import typing

import numpy as np
import pandas as pd

from morpheus.common import FileTypes
from morpheus.common import determine_file_type
from morpheus.io.deserializers import read_file_to_df
from morpheus.stages.input.autoencoder_source_stage import AutoencoderSourceStage

logger = logging.getLogger(__name__)


class CloudTrailSourceStage(AutoencoderSourceStage):
    """
    Load messages from a CloudTrail directory.

    """

    @property
    def name(self) -> str:
        return "from-cloudtrail"

    @property
    def input_count(self) -> int:
        """Return None for no max intput count"""
        return self._input_count

    def supports_cpp_node(self):
        return False

    def get_match_pattern(self, glob_split):
        """Return a file match pattern"""
        dir_to_watch = os.path.dirname(glob_split[0])
        match_pattern = self._input_glob.replace(dir_to_watch + "/", "", 1)

        return match_pattern

    @staticmethod
    def read_file(filename: str, file_type: FileTypes) -> pd.DataFrame:
        """
        Reads a file into a dataframe.

        Parameters
        ----------
        filename : str
            Path to a file to read.
        file_type : `morpheus.common.FileTypes`
            What type of file to read. Leave as Auto to auto detect based on the file extension.

        Returns
        -------
        pandas.DataFrame
            The parsed dataframe.

        Raises
        ------
        RuntimeError
            If an unsupported file type is detected.
        """

        df = read_file_to_df(filename, file_type, df_type="pandas")

        # If reading the file only produced one line and we are a JSON file, try loading structured file
        if (determine_file_type(filename) == FileTypes.JSON and len(df) == 1 and list(df) == ["Records"]):

            # Reread with lines=False
            df = read_file_to_df(filename, file_type, df_type="pandas", parser_kwargs={"lines": False})

            # Normalize
            df = pd.json_normalize(df['Records'])

        return df

    @staticmethod
    def cleanup_df(df: pd.DataFrame, feature_columns: typing.List[str]):
        """
        This function does clean up certain columns in the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe for columns cleanup.
        feature_columns : typing.List[str]
            Only the columns that are present in the feature columns will be preserved in the dataframe
            if feature columns are supplied..

        Returns
        -------
        df : typing.List[pd.DataFrame]
            Clean dataframe.
        """

        # Replace all the dots in column names
        df.columns = df.columns.str.replace('.', '', regex=False)

        df["event_dt"] = pd.to_datetime(df["eventTime"])

        def remove_null(x):

            if isinstance(x, list):
                if isinstance(x[0], dict):
                    key = list(x[0].keys())
                    return x[0][key[0]]
            return x

        def clean_column(cloudtrail_df):

            col_name = 'requestParametersownersSetitems'
            if (col_name in cloudtrail_df):
                cloudtrail_df[col_name] = cloudtrail_df[col_name].apply(lambda x: remove_null(x))
            return cloudtrail_df

        # Drop any unneeded columns if specified
        if (feature_columns is not None):
            df.drop(columns=df.columns.difference(feature_columns), inplace=True)

        # Reorder columns to be the same
        # df = df[pd.Index(feature_columns).intersection(df.columns)]

        # Convert a numerical account ID into a string
        if ("userIdentityaccountId" in df and df["userIdentityaccountId"].dtype != np.dtype('O')):
            df['userIdentityaccountId'] = 'Account-' + df['userIdentityaccountId'].astype(str)

        df = clean_column(df)

        return df

    @staticmethod
    def files_to_dfs_per_user(x: typing.List[str],
                              userid_column_name: str,
                              feature_columns: typing.List[str],
                              userid_filter: str = None,
                              repeat_count: int = 1) -> typing.Dict[str, pd.DataFrame]:
        """
        After loading the input batch of CloudTrail logs into a dataframe, this method builds a dataframe
        for each set of userid rows in accordance with the specified filter condition.

        Parameters
        ----------
        x : typing.List[str]
            List of messages.
        userid_column_name : str
            Name of the column used for categorization.
        feature_columns : typing.List[str]
            Feature column names.
        userid_filter : str
            Only rows with the supplied userid are filtered.
        repeat_count : str
            Number of times the given rows should be repeated.

        Returns
        -------
        df_per_user  : typing.Dict[str, pd.DataFrame]
            Dataframe per userid.
        """

        # Using pandas to parse nested JSON until cuDF adds support
        # https://github.com/rapidsai/cudf/issues/8827
        dfs = []
        for file in x:
            df = CloudTrailSourceStage.read_file(file, FileTypes.Auto)
            df = CloudTrailSourceStage.cleanup_df(df, feature_columns)
            dfs = dfs + CloudTrailSourceStage.repeat_df(df, repeat_count)

        df_per_user = CloudTrailSourceStage.batch_user_split(dfs, userid_column_name, userid_filter)

        return df_per_user
