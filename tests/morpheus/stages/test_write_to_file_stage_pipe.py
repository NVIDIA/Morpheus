#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import types

import pandas as pd
import pytest

from _utils.dataset_manager import DatasetManager
from morpheus.common import FileTypes
from morpheus.io.deserializers import read_file_to_df
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage


@pytest.mark.parametrize("output_file", ["/tmp/output.json", "/tmp/output.csv", "/tmp/output.parquet"])
@pytest.mark.gpu_and_cpu_mode
def test_write_to_file_stage_pipe(config,
                                  df_pkg: types.ModuleType,
                                  dataset: DatasetManager,
                                  output_file: str,
                                  execution_mode: str) -> None:
    """
    Test WriteToFileStage with different output formats (JSON, CSV, Parquet)
    """

    filter_probs_df = dataset['filter_probs.csv']
    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(SerializeStage(config))
    pipe.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))
    pipe.run()

    if output_file.endswith(".json"):
        output_df = pd.read_json(output_file, lines=True)
        dataset.assert_compare_df(filter_probs_df, output_df)

    elif output_file.endswith(".csv"):
        # The output data will contain an additional id column that we will need to slice off
        output_df = df_pkg.read_csv(output_file).iloc[:, 1:]
        dataset.assert_compare_df(filter_probs_df, output_df)

    elif output_file.endswith(".parquet"):
        output_df = read_file_to_df(file_name=output_file, file_type=FileTypes.PARQUET)
        # The c++ WriteToFileStage will add an additional index column to the output
        if execution_mode == "GPU":
            output_df = output_df.iloc[:, 1:]
        assert output_df.values.tolist() == filter_probs_df.values.tolist()
