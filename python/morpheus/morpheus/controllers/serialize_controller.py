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

import copy
import re
import typing

from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta


class SerializeController:
    """
    Controller class for converting data to JSON lines format with customizable column selection and exclusion.

    Parameters
    ----------
    include : typing.List[str]
        List of columns to include.
    exclude : typing.List[str]
        List of columns to exclude.
    fixed_columns : bool
        Flag to indicate whether columns should be fixed.
    """

    def __init__(self, include: typing.List[str], exclude: typing.List[str], fixed_columns: bool):
        self._include_columns = copy.copy(include)
        self._exclude_columns = copy.copy(exclude)
        self._fixed_columns = fixed_columns
        self._columns = None

    @property
    def include_columns(self):
        """
        Get the list of included columns.
        """
        return self._include_columns

    @property
    def exclude_columns(self):
        """
        Get the list of excluded columns.
        """
        return self._exclude_columns

    @property
    def fixed_columns(self):
        """
        Get the flag indicating whether columns are fixed.
        """
        return self._fixed_columns

    def convert_to_df(self,
                      msg: ControlMessage,
                      include_columns: typing.Pattern,
                      exclude_columns: typing.List[typing.Pattern]):
        """
        Converts dataframe to entries to JSON lines.

        Parameters
        ----------
        msg :`morpheus.messages.ControlMessage`
            ControlMessage instance that contains data.
        include_columns : typing.Pattern
            Columns that are required send to downstream stage.
        exclude_columns : typing.List[typing.Pattern]
            Columns that are not required send to downstream stage.

        """

        if self._fixed_columns and self._columns is not None:
            columns = self._columns
        else:
            columns: typing.List[str] = []

            # Minimize access to x.meta.df
            df_columns = list(msg.payload().get_column_names())

            # First build up list of included. If no include regex is specified, select all
            if (include_columns is None):
                columns = df_columns
            else:
                columns = [y for y in df_columns if include_columns.match(y)]

            # Now remove by the ignore
            for test in exclude_columns:
                columns = [y for y in columns if not test.match(y)]

            self._columns = columns

        # Get metadata from columns
        df = msg.payload().get_data(columns)

        return MessageMeta(df=df)

    def get_include_col_pattern(self):
        """
        Get the compiled pattern for include columns.

        Returns
        -------
        typing.Pattern
            The compiled pattern for include columns.
        """

        include_columns = None

        if (self._include_columns is not None and len(self._include_columns) > 0):
            include_columns = re.compile(f"({'|'.join(self._include_columns)})")

        return include_columns

    def get_exclude_col_pattern(self):
        """
        Get the list of compiled patterns for exclude columns.

        Returns
        -------
        typing.List[typing.Pattern]
            The list of compiled patterns for exclude columns.
        """
        exclude_columns = [re.compile(x) for x in self._exclude_columns]

        return exclude_columns
