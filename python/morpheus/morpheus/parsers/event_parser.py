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
"""Abstract class for all event log parsers."""

import logging
from abc import ABC
from abc import abstractmethod

import yaml

from morpheus.utils.type_aliases import DataFrameType
from morpheus.utils.type_aliases import SeriesType
from morpheus.utils.type_utils import get_df_pkg_from_obj

log = logging.getLogger(__name__)


class EventParser(ABC):
    """
    This is an abstract class for all event log parsers.

    Parameters
    ----------
    columns: set[str]
        Event column names
    event_name: str
        Event name
    """

    def __init__(self, columns: set[str], event_name: str):
        self._columns = columns
        self._event_name = event_name

    @property
    def columns(self):
        """
        List of columns that are being processed.

        Returns
        -------
        set[str]
            Event column names
        """
        return self._columns

    @property
    def event_name(self):
        """
        Event name define type of logs that are being processed.

        Returns
        -------
        str
            Event name
        """
        return self._event_name

    @abstractmethod
    def parse(self, text: SeriesType) -> SeriesType:
        """
        Abstract method 'parse' triggers the parsing functionality. Subclasses are required to implement
        and execute any parsing pre-processing steps.
        """
        log.info("Begin parsing of dataframe")
        pass

    def parse_raw_event(self, text: SeriesType, event_regex: dict[str, str]) -> DataFrameType:
        """
        Processes parsing of a specific type of raw event records received as a dataframe.

        Parameters
        ----------
        text : SeriesType
            Raw event log text to be parsed.
        event_regex: typing.Dict[str, str]
            Required regular expressions for a given event type.

        Returns
        -------
        DataFrameType
            Parsed logs dataframe
        """
        log.debug("Parsing raw events. Event type: %s", self.event_name)

        df_pkg = get_df_pkg_from_obj(text)
        parsed_gdf = df_pkg.DataFrame({col: [""] for col in self.columns})
        parsed_gdf = parsed_gdf[:0]
        event_specific_columns = event_regex.keys()
        # Applies regex pattern for each expected output column to raw data
        for col in event_specific_columns:
            regex_pattern = event_regex.get(col)
            extracted_gdf = text.str.extract(regex_pattern).reset_index()
            if not extracted_gdf.empty:
                parsed_gdf[col] = extracted_gdf[0]

        remaining_columns = list(self.columns - event_specific_columns)
        # Fill remaining columns with empty.
        for col in remaining_columns:
            parsed_gdf[col] = ""

        return parsed_gdf

    def _load_regex_yaml(self, yaml_file) -> dict[str, str]:
        """Returns a dictionary of event regexes contained in the given yaml file."""
        with open(yaml_file, encoding='UTF-8') as yaml_file_h:
            regex_dict = yaml.safe_load(yaml_file_h)
        return regex_dict
