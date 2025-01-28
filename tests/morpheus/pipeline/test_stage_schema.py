#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from _utils.stages.split_stage import SplitStage
from morpheus.messages import MessageMeta
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage


# Fixtures cannot be used directly as paramertize values, but we can fetch them by name
@pytest.mark.parametrize("stage_fixture_name,num_inputs,num_outputs",
                         [("in_mem_source_stage", 0, 1), ("in_mem_multi_source_stage", 0, 3), ("stage", 1, 1),
                          ("split_stage", 1, 2)])
def test_constructor(request: pytest.FixtureRequest, stage_fixture_name: str, num_inputs: int, num_outputs: int):
    stage = request.getfixturevalue(stage_fixture_name)
    schema = StageSchema(stage)
    assert len(schema.input_schemas) == num_inputs
    assert len(schema.output_schemas) == num_outputs


def test_single_port_input_schemas(stage: DeserializeStage):
    schema = StageSchema(stage)
    assert len(schema.input_schemas) == 1

    port_schema = schema.input_schemas[0]
    assert port_schema.get_type() is MessageMeta

    assert schema.input_schema is port_schema


def test_single_port_input_types(stage: DeserializeStage):
    schema = StageSchema(stage)
    assert len(schema.input_types) == 1

    assert schema.input_types[0] is MessageMeta
    assert schema.input_type is MessageMeta


def test_single_port_output_schemas(in_mem_source_stage: InMemorySourceStage):
    schema = StageSchema(in_mem_source_stage)
    in_mem_source_stage.compute_schema(schema)
    assert len(schema.output_schemas) == 1

    port_schema = schema.output_schemas[0]
    assert port_schema.get_type() is MessageMeta

    assert schema.output_schema is port_schema


def test_multi_port_output_schemas(split_stage: SplitStage):
    schema = StageSchema(split_stage)
    split_stage.compute_schema(schema)
    assert len(schema.output_schemas) == 2

    for port_schema in schema.output_schemas:
        assert port_schema.get_type() is MessageMeta


@pytest.mark.parametrize("stage_fixture_name", ["split_stage"])
def test_output_schema_multi_error(request: pytest.FixtureRequest, stage_fixture_name: str):
    """
    Test confirms that the output_schema property raises an error when there are multiple output schemas
    """
    stage = request.getfixturevalue(stage_fixture_name)
    schema = StageSchema(stage)
    stage.compute_schema(schema)
    assert len(schema.output_schemas) > 1

    with pytest.raises(AssertionError):
        schema.output_schema  # pylint: disable=pointless-statement


@pytest.mark.parametrize(
    "stage_fixture_name",
    ["in_mem_source_stage", "in_mem_multi_source_stage", "stage", "split_stage", "multi_pass_thru_stage"])
def test_complete(request: pytest.FixtureRequest, stage_fixture_name: str):
    stage = request.getfixturevalue(stage_fixture_name)
    schema = StageSchema(stage)
    stage.compute_schema(schema)

    for port_schema in schema.output_schemas:
        assert not port_schema.is_complete()

    schema._complete()

    for port_schema in schema.output_schemas:
        assert port_schema.is_complete()
