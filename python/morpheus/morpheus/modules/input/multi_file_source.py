# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
import time
import typing

import fsspec
import mrc
from pydantic import ValidationError

from morpheus.modules.schemas.multi_file_source_schema import MultiFileSourceSchema
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(f"morpheus.{__name__}")

MultiFileSourceLoaderFactory = ModuleLoaderFactory("multi_file_source", "morpheus", MultiFileSourceSchema)


def expand_paths_simple(filenames: typing.List[str]) -> typing.List[str]:
    """
    Expand to glob all files in any directories in the input filenames,
    provided they actually exist.

    Parameters
    ----------
    filenames : typing.List[str]
        A list of filenames or directories to expand.

    Returns
    -------
    typing.List[str]
        A list of filenames with directories expanded to glob patterns.

    Examples
    --------
    >>> expand_paths_simple(['/path/to/dir'])
    ['/path/to/dir/*']

    Notes
    -----
    If a filename in the list already contains a wildcard character (* or ?),
    it is appended to the output list as is.
    """
    updated_list = []
    fs_spec = fsspec.filesystem(protocol='file')
    for file_name in filenames:
        if '*' in file_name or '?' in file_name:
            updated_list.append(file_name)
            continue

        if (not fs_spec.exists(file_name)):
            updated_list.append(file_name)
            continue

        if fs_spec.isdir(file_name):
            updated_list.append(f"{file_name}/*")
        else:
            updated_list.append(file_name)

    return updated_list


@register_module("multi_file_source", "morpheus")
def _multi_file_source(builder: mrc.Builder):
    """
    Creates a file source module for the Morpheus builder. This module reads files
    from a specified source and processes them accordingly.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus builder instance to attach this module to.

    Raises
    ------
    ValueError
        If the source_config does not contain a list of filenames.

    Notes
    -----
    - The module configuration parameters include:
        - 'filenames': List of filenames or wildcard paths to read from.
        - 'watch_dir': Boolean indicating whether to watch the directory for changes.
        - 'watch_interval': Time interval (in seconds) for watching the directory.
        - 'batch_size': The number of files to process in a batch.
    """
    module_config = builder.get_current_module_config()
    source_config = module_config.get('source_config', {})

    try:
        validated_config = MultiFileSourceSchema(**source_config)
    except ValidationError as e:
        # Format the error message for better readability
        error_messages = '; '.join([f"{error['loc'][0]}: {error['msg']}" for error in e.errors()])
        log_error_message = f"Invalid configuration for file_content_extractor: {error_messages}"
        logger.error(log_error_message)

        raise

    filenames = expand_paths_simple(validated_config.filenames)
    watch_dir = validated_config.watch_dir
    watch_interval = validated_config.watch_interval
    batch_size = validated_config.batch_size

    def polling_generate_frames_fsspec():
        files_seen = set()

        while True:
            start_time = time.monotonic()
            next_update_epoch = start_time + watch_interval

            if not filenames:
                # Log warning or handle the case where filenames is None or empty
                logger.warning("No filenames provided. Skipping iteration.")
                time.sleep(watch_interval)
                continue

            files = fsspec.open_files(filenames)

            new_files = [file for file in files if file.full_name not in files_seen]

            # Update files_seen with the new set of files
            files_seen.update(file.full_name for file in new_files)

            # Process new files in batches
            batch = []
            for file in new_files:
                batch.append(file)
                if len(batch) >= batch_size or time.monotonic() - start_time >= 1.0:
                    yield fsspec.core.OpenFiles(batch, fs=files.fs)
                    batch = []
                    start_time = time.monotonic()

            # Yield remaining files if any
            if batch:
                yield fsspec.core.OpenFiles(batch, fs=files.fs)

            # Sleep until the next update epoch
            sleep_duration = next_update_epoch - time.monotonic()
            if sleep_duration > 0:
                time.sleep(sleep_duration)

    def generate_frames_fsspec():
        # Check if filenames is None or empty
        if (not filenames):
            logger.warning("Multi-file-source was provided with no filenames for processing this is probably not what"
                           "you want")
            return

        files = fsspec.open_files(filenames)

        # Check if the provided filenames resulted in any files being opened
        if len(files) == 0:
            logger.warning("Multi-file-source did not match any of the provided filter strings: %s. %s",
                           filenames,
                           "This is probably not what you want.")
            return

        logger.info("File source exhausted, discovered %s files.", len(files))

        yield files

    if (watch_dir):
        node = builder.make_source("multi_file_source", polling_generate_frames_fsspec)
    else:
        node = builder.make_source("multi_file_source", generate_frames_fsspec)

    builder.register_module_output("output", node)
