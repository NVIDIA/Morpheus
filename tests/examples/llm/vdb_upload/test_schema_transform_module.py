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

import types

import pytest

import cudf

from _utils import assert_results
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage


@pytest.mark.parametrize("num_select, num_renames", [(1, 0), (0, 1), (1, 1), (6, 6), (13, 10), (10, 13)])
def test_schema_transform_module(num_select,
                                 num_renames,
                                 config: Config,
                                 import_schema_transform_module: types.ModuleType):
    # Generate the DataFrame columns for select and rename
    select_columns = [f'select_{i}' for i in range(num_select)]
    rename_columns = [f'rename_from_{i}' for i in range(num_renames)]

    # Generate the DataFrame
    df_data = {col: range(10) for col in select_columns}
    df_data.update({col: range(10) for col in rename_columns})
    df = cudf.DataFrame(df_data)

    # Generate the expected DataFrame
    expected_data = {col: range(10) for col in select_columns}
    expected_data.update({f'rename_to_{i}': range(10) for i in range(num_renames)})
    df_expected = cudf.DataFrame(expected_data)

    # Generate the schema transform configuration
    transform_config = {
        "schema_transform_config": {
            col: {
                "dtype": "int", "op_type": "select"
            }
            for col in select_columns
        }
    }
    transform_config["schema_transform_config"].update({
        f'rename_to_{i}': {
            "from": f'rename_from_{i}', "dtype": "int", "op_type": "rename"
        }
        for i in range(num_renames)
    })

    schema_module_loader = import_schema_transform_module.SchemaTransformLoaderFactory.get_instance(
        "schema_transform", module_config=transform_config)

    # Set up the pipeline
    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [df]))
    pipe.add_stage(
        LinearModulesStage(config,
                           schema_module_loader,
                           input_type=MessageMeta,
                           output_type=MessageMeta,
                           input_port_name="input",
                           output_port_name="output"))

    comp_stage = pipe.add_stage(CompareDataFrameStage(config, compare_df=df_expected))
    pipe.run()

    assert_results(comp_stage.get_results())
