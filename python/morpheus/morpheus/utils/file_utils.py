# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""File utilities for Morpheus"""

import os
import re
from datetime import datetime
from datetime import timezone

import fsspec

import morpheus


def get_data_file_path(data_filename: str) -> str:
    """
    Get data file path. Also handles paths relative to Morpheus root.

    Parameters
    ----------
    data_filename : str
        Absolute or relative path of data file.

    Returns
    -------
    str
        Data file path.
    """
    # First check if the path is relative
    if (os.path.isabs(data_filename)):
        # Already absolute, nothing to do
        return data_filename

    # See if the file exists.
    does_exist = os.path.exists(data_filename)

    if (not does_exist):
        # If it doesn't exist, then try to make it relative to the morpheus library root
        morpheus_root = os.path.dirname(morpheus.__file__)

        value_abs_to_root = os.path.join(morpheus_root, data_filename)

        # If the file relative to our package exists, use that instead
        if (os.path.exists(value_abs_to_root)):
            return value_abs_to_root

    return data_filename


def load_labels_file(labels_filename: str) -> list[str]:
    """
    Get list of labels from file.

    Parameters
    ----------
    labels_filename : str
        Labels file path

    Returns
    -------
    list[str]
        List of labels
    """
    with open(labels_filename, "r", encoding='UTF-8') as fh:
        return [x.strip() for x in fh.readlines()]


def date_extractor(file_object: fsspec.core.OpenFile, filename_regex: re.Pattern):
    """
    Date is extracted from a file name using a specified regex pattern by this function.
    If there is no match with the pattern, extracts the modified or creation timestamp from the file.

    Parameters
    ----------
    file_object : fsspec.core.OpenFile
        File object
    filename_regex : re.Pattern
        Filename regex.

    Returns
    -------
    datetime
        Extracted timestamp
    """
    if not isinstance(file_object, fsspec.core.OpenFile):
        raise ValueError("file_object must be an instance of fsspec.core.OpenFile")

    file_path = file_object.path

    # Match regex with the pathname
    match = filename_regex.search(file_path)

    if match:
        groups = match.groupdict()

        # Convert the microsecond value if present
        if groups.get("microsecond"):
            groups["microsecond"] = min(int(float(groups["microsecond"]) * 1000000), 999999)

        # Filter out any None values and convert the rest to integers
        groups = {key: int(value) for key, value in groups.items() if value and key != "zulu"}

        # Assign timezone
        groups["tzinfo"] = timezone.utc

        ts_object = datetime(**groups)
    else:
        # Fallback to the file modified time
        ts_object = file_object.fs.modified(file_object.path)

        # Set the timezone to the current system's timezone
        ts_object = ts_object.replace(tzinfo=datetime.now().astimezone().tzinfo)

    return ts_object
