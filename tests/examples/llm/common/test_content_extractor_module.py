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
import random
import shutil
import string
import tempfile
import types
import uuid
from functools import partial
from typing import Callable
from typing import Dict
from typing import Generator
from typing import List

import fsspec.core
import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from morpheus.stages.input.in_memory_data_generation_stage import InMemoryDataGenStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage

logger = logging.getLogger(f"morpheus.{__name__}")


class TempCSVFiles:

    def __init__(self, num_files: int, columns: Dict[str, Callable[[], any]]):
        self.temp_dir = None
        self.temp_files = []
        self.num_files = num_files
        self.columns = columns

        self._create_temp_dir_and_files()

    def _create_temp_dir_and_files(self):
        # Create a temporary directory
        self.temp_dir = os.path.join(tempfile.gettempdir(), uuid.uuid4().hex)
        os.makedirs(self.temp_dir, exist_ok=True)

        for _ in range(self.num_files):
            # Create a random filename within the temp directory
            file_path = os.path.join(self.temp_dir, f"{uuid.uuid4().hex}.csv")

            # Generate deterministic CSV data based on the specified columns
            data = {col_name: col_func() for col_name, col_func in self.columns.items()}
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)

            # Store the file path for later use
            self.temp_files.append(file_path)

    def __enter__(self):
        return self.temp_files

    def __exit__(self, exc_type, exc_value, traceback):
        # Cleanup the temporary directory and its contents
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


# Define a generator function that uses TempCSVFiles to generate CSV file paths
def csv_file_generator(csv_files: List[str], batch_size: int) -> Generator[List[fsspec.core.OpenFile], None, None]:
    # Create TempCSVFiles instance without using 'with' statement
    open_files = fsspec.open_files(csv_files.temp_files)
    for i in range(0, len(open_files), batch_size):
        yield open_files[i:i + batch_size]


def generate_random_string(length: int) -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


@pytest.mark.parametrize("data_len, num_rows_per_file, batch_size",
                         [(40, 5, 2), (51, 3, 1), (150, 10, 5), (500, 3, 2), (1000, 5, 3), (50, 10, 2), (100, 20, 3),
                          (50, 5, 1), (100, 10, 1), (49, 5, 2), (99, 5, 2), (60, 7, 2), (120, 6, 3), (1000, 50, 10),
                          (2000, 100, 20)])
def test_content_extractor_module(data_len,
                                  num_rows_per_file,
                                  batch_size,
                                  config: Config,
                                  import_content_extractor_module: types.ModuleType):
    chunk_size = 50
    chunk_overlap = 10
    # Text splitter handles things a bit differently on evenly divisible boundaries
    chunk_boundary_size = (chunk_size - chunk_overlap) if (data_len > chunk_size) else chunk_size
    module_config = {
        "batch_size": batch_size,
        "chunk_size": 512,
        "chunk_overlap": 51,
        "converters_meta": {
            "csv": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "text_column_names": ["some_column"],
            }
        },
    }
    content_extractor_loader = import_content_extractor_module.ContentExtractorLoaderFactory.get_instance(
        "content_extractor", module_config=module_config)

    temp_csv_files = TempCSVFiles(
        num_files=5,
        columns={'some_column': lambda: [generate_random_string(data_len) for _ in range(num_rows_per_file)]})
    file_generator = partial(csv_file_generator, temp_csv_files, batch_size=1)

    pipe = LinearPipeline(config)
    pipe.set_source(InMemoryDataGenStage(config, file_generator, output_data_type=List[fsspec.core.OpenFile]))
    pipe.add_stage(
        LinearModulesStage(config,
                           content_extractor_loader,
                           input_type=List[fsspec.core.OpenFile],
                           output_type=MessageMeta,
                           input_port_name="input",
                           output_port_name="output"))
    sink_stage = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    expected_columns = ["title", "source", "summary", "content"]
    for message in sink_stage.get_messages():
        output = message.df
        assert set(expected_columns) == set(output.columns)
        assert output.shape == (num_rows_per_file * ((data_len // chunk_boundary_size) +
                                                     (1 if data_len % chunk_boundary_size else 0)),
                                4)
