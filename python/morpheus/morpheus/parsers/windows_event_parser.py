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
import os
import typing

import morpheus
from morpheus.parsers.event_parser import EventParser
from morpheus.utils.type_aliases import DataFrameType
from morpheus.utils.type_aliases import SeriesType
from morpheus.utils.type_utils import get_df_pkg_from_obj

log = logging.getLogger(__name__)


class WindowsEventParser(EventParser):
    """
    This is class parses windows event logs.

    Parameters
    ----------
    interested_eventcodes: typing.Set[int]
        Set of interested codes to parse
    """
    EVENT_NAME = "windows-event"

    def __init__(self, interested_eventcodes=None):
        regex_filepath = os.path.join(morpheus.DATA_DIR, "windows_event_regex.yaml")
        self._interested_eventcodes = interested_eventcodes
        self._event_regex = self._load_regex_yaml(regex_filepath)
        EventParser.__init__(self, self.get_columns(), self.EVENT_NAME)

    def parse(self, text: SeriesType) -> DataFrameType:
        """Parses the Windows raw event.

        Parameters
        ----------
        text : SeriesType
            Raw event log text to be parsed

        Returns
        -------
        DataFrameType
            Parsed logs dataframe
        """
        # Clean raw data to be consistent.
        text = self.clean_raw_data(text)
        output_chunks = []
        for eventcode in self._event_regex.keys():
            pattern = f"eventcode={eventcode}"
            # input_chunk = self.filter_by_pattern(dataframe, raw_column, pattern)
            input_chunk = text[text.str.contains(pattern)]
            if not input_chunk.empty:
                temp = self.parse_raw_event(input_chunk, self._event_regex[eventcode])
                if not temp.empty:
                    output_chunks.append(temp)

        df_pkg = get_df_pkg_from_obj(text)
        parsed_dataframe = df_pkg.concat(output_chunks)
        # Replace null values with empty.
        parsed_dataframe = parsed_dataframe.fillna("")
        return parsed_dataframe

    def clean_raw_data(self, text: SeriesType) -> SeriesType:
        """
        Lower casing and replacing escape characters.

        Parameters
        ----------
        text : SeriesType
            Raw event log text to be clean

        Returns
        -------
        SeriesType
            Clean raw event log text
        """
        text = (text.str.lower().str.replace("\\\\t", "").str.replace("\\\\r", "").str.replace("\\\\n", "|"))
        return text

    def _load_regex_yaml(self, yaml_file):
        event_regex = EventParser._load_regex_yaml(self, yaml_file)
        if self._interested_eventcodes is not None:
            for eventcode in self._interested_eventcodes:
                required_event_regex = {}
                if eventcode not in event_regex:
                    raise KeyError(f"Regex for eventcode {eventcode} is not available in the config file. "
                                   f"Please choose from {list(event_regex.keys())}")
                required_event_regex[eventcode] = event_regex[eventcode]
            return required_event_regex
        return event_regex

    def get_columns(self) -> typing.Set[str]:
        """
        Get columns of windows event codes.

        Returns
        -------
        typing.Set[str]
            Columns of all configured eventcodes, if no interested eventcodes specified.
        """
        columns = set()
        for key in self._event_regex.keys():
            for column in self._event_regex[key].keys():
                columns.add(column)
        return columns
