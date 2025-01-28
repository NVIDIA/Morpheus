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

import collections
import functools
import inspect
import typing
from unittest import mock

import mrc
import pandas as pd
import pytest
from mrc import Subscription

import cudf

from _utils import assert_results
from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.stage_decorator import ComputeSchemaType
from morpheus.pipeline.stage_decorator import PreAllocatedWrappedFunctionStage
from morpheus.pipeline.stage_decorator import WrappedFunctionSourceStage
from morpheus.pipeline.stage_decorator import WrappedFunctionStage
from morpheus.pipeline.stage_decorator import _fn_receives_subscription
from morpheus.pipeline.stage_decorator import source
from morpheus.pipeline.stage_decorator import stage
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.utils.type_aliases import DataFrameType


def _get_annotation(type_: type, generator_type: type) -> type:
    if generator_type is not None:
        if generator_type in (typing.Generator, collections.abc.Generator):
            annotation = generator_type[type_, None, None]
        else:
            annotation = generator_type[type_]
    else:
        annotation = type_

    return annotation


def _mk_compute_schema_fn(return_type: type) -> ComputeSchemaType:
    return lambda schema: schema.output_schema.set_type(return_type)


@pytest.mark.parametrize("generator_type",
                         [None, typing.Iterator, typing.Generator, collections.abc.Iterator, collections.abc.Generator])
@pytest.mark.parametrize("return_type, is_prealloc", [(pd.DataFrame, True), (cudf.DataFrame, True), (MessageMeta, True),
                                                      (float, False)])
def test_wrapped_function_source_stage_constructor(config: Config,
                                                   generator_type: type,
                                                   return_type: type,
                                                   is_prealloc: bool):
    return_annotation = _get_annotation(return_type, generator_type)

    def test_source_gen() -> return_annotation:
        yield None

    if is_prealloc:
        source_cls = PreAllocatedWrappedFunctionStage
    else:
        source_cls = WrappedFunctionSourceStage

    mock_compute_schema_fn = mock.MagicMock()
    mock_compute_schema_fn.side_effect = _mk_compute_schema_fn(return_type)

    source_stage = source_cls(config,
                              name="unittest-source",
                              gen_fn=test_source_gen,
                              compute_schema_fn=mock_compute_schema_fn)

    assert isinstance(source_stage, WrappedFunctionSourceStage)
    assert is_prealloc == isinstance(source_stage, PreAllocatedWrappedFunctionStage)

    # check the output type
    schema = StageSchema(source_stage)
    source_stage.compute_schema(schema)
    assert schema.output_schema.get_type() is return_type
    mock_compute_schema_fn.assert_called_once_with(schema)


@pytest.mark.parametrize("src_cls", [WrappedFunctionSourceStage, PreAllocatedWrappedFunctionStage])
def test_wrapped_function_source_stage_not_generator_error(config: Config, src_cls: type):

    def test_source_gen() -> MessageMeta:
        return MessageMeta(cudf.DataFrame())

    with pytest.raises(ValueError):
        src_cls(config,
                name="unittest-source",
                gen_fn=test_source_gen,
                compute_schema_fn=_mk_compute_schema_fn(MessageMeta))


@pytest.mark.parametrize("generator_type",
                         [None, typing.Iterator, typing.Generator, collections.abc.Iterator, collections.abc.Generator])
@pytest.mark.parametrize("return_type, is_prealloc", [(pd.DataFrame, True), (cudf.DataFrame, True), (MessageMeta, True),
                                                      (float, False)])
def test_source_decorator(config: Config, generator_type: type, return_type: type, is_prealloc: bool):
    return_annotation = _get_annotation(return_type, generator_type)

    @source
    def test_source_gen() -> return_annotation:
        yield None

    source_stage = test_source_gen(config)  # pylint: disable=too-many-function-args

    assert isinstance(source_stage, WrappedFunctionSourceStage)
    assert is_prealloc == isinstance(source_stage, PreAllocatedWrappedFunctionStage)

    # check the output type
    schema = StageSchema(source_stage)
    source_stage.compute_schema(schema)  # pylint: disable=no-member
    assert schema.output_schema.get_type() is return_type


def test_source_decorator_name(config: Config):

    @source
    def test_source_gen(value: int) -> int:
        yield value

    source_stage = test_source_gen(config, value=5)  # pylint: disable=redundant-keyword-arg
    assert source_stage.name == 'test_source_gen'  # pylint: disable=no-member


def test_source_decorator_explicit_name(config: Config):

    @source(name="source_gen")
    def test_source_gen(value: int) -> int:
        yield value

    source_stage = test_source_gen(config, value=5)  # pylint: disable=redundant-keyword-arg
    assert source_stage.name == 'source_gen'  # pylint: disable=no-member


def test_source_decorator_explicit_compute_schema(config: Config):
    mock_compute_schema_fn = mock.MagicMock()
    mock_compute_schema_fn.side_effect = _mk_compute_schema_fn(int)

    @source(compute_schema_fn=mock_compute_schema_fn)
    def test_source_gen(value: int) -> int:
        yield value

    source_stage = test_source_gen(config, value=5)  # pylint: disable=redundant-keyword-arg
    schema = StageSchema(source_stage)
    source_stage.compute_schema(schema)  # pylint: disable=no-member
    assert schema.output_schema.get_type() is int
    mock_compute_schema_fn.assert_called_once_with(schema)


def test_source_decorator_no_annoation_error(config: Config):

    @source
    def test_source_gen():
        yield None

    with pytest.raises(ValueError):
        test_source_gen(config)  # pylint: disable=too-many-function-args


def test_not_generator_error(config: Config):

    @source
    def test_fn() -> int:
        return 5

    with pytest.raises(ValueError):
        test_fn(config)  # pylint: disable=too-many-function-args


def test_source_stage_arg_no_value_error(config: Config):

    @source
    def test_source_gen(value: int) -> int:
        yield value

    with pytest.raises(ValueError):
        test_source_gen(config)


@pytest.mark.parametrize("accept_type, return_type",
                         [(pd.DataFrame, MessageMeta), (int, int), (MessageMeta, MessageMeta), (typing.Any, bool),
                          (typing.Union[float, int], float), (float, typing.Any), (typing.Any, float),
                          (typing.Any, typing.Any)])
def test_wrapped_function_stage_constructor(config: Config, accept_type: type, return_type: type):
    wrapped_stage = WrappedFunctionStage(config,
                                         name="unittest-stage",
                                         on_data_fn=lambda x: x,
                                         accept_type=accept_type,
                                         compute_schema_fn=_mk_compute_schema_fn(return_type))

    assert isinstance(wrapped_stage, WrappedFunctionStage)
    assert wrapped_stage.accepted_types() == (accept_type, )


@pytest.mark.parametrize("accept_type, return_type",
                         [(pd.DataFrame, MessageMeta), (int, int), (MessageMeta, MessageMeta), (typing.Any, bool),
                          (typing.Union[float, int], float), (float, float), (typing.Any, float),
                          (typing.Any, typing.Any)])
def test_wrapped_function_stage_output_types(config: Config, accept_type: type, return_type: type):
    # For non-source types we need an upstream before we can check the compute_schema method outside of a pipeline

    mock_compute_schema_fn = mock.MagicMock()
    mock_compute_schema_fn.side_effect = _mk_compute_schema_fn(return_type)

    wrapped_stage = WrappedFunctionStage(config,
                                         name="unittest-stage",
                                         on_data_fn=lambda x: x,
                                         accept_type=accept_type,
                                         compute_schema_fn=mock_compute_schema_fn)

    def source_fn():
        yield None

    upstream = WrappedFunctionSourceStage(config,
                                          name="source_fn",
                                          gen_fn=source_fn,
                                          compute_schema_fn=_mk_compute_schema_fn(accept_type))

    pipe = LinearPipeline(config)
    pipe.set_source(upstream)
    pipe.add_stage(wrapped_stage)
    pipe.build()
    mock_compute_schema_fn.assert_called_once()  # pipe.build() will call wrapped_stage.compute_schema()

    schema = StageSchema(wrapped_stage)
    wrapped_stage.compute_schema(schema)
    assert schema.output_schema.get_type() is return_type


def test_wrapped_function_stage_name(config: Config):

    def multiplier(message: MessageMeta, column: str, value: int | float) -> MessageMeta:
        with message.mutable_dataframe() as df:
            df[column] = df[column] * value

        return message

    wrapped_stage = WrappedFunctionStage(config,
                                         name="multiplier",
                                         on_data_fn=functools.partial(multiplier, column='v2', value=5),
                                         accept_type=MessageMeta,
                                         compute_schema_fn=_mk_compute_schema_fn(MessageMeta))
    assert wrapped_stage.name == 'multiplier'


@pytest.mark.parametrize("needed_columns",
                         [None, {
                             'result': TypeId.INT64
                         }, {
                             'a': TypeId.INT64, 'b': TypeId.FLOAT32, 'c': TypeId.STRING
                         }])
def test_wrapped_function_stage_needed_columns(config: Config, needed_columns: dict[str, TypeId]):

    def test_fn(message: MessageMeta) -> MessageMeta:
        return message

    wrapped_stage = WrappedFunctionStage(config,
                                         name="unittest-stage",
                                         on_data_fn=test_fn,
                                         accept_type=MessageMeta,
                                         compute_schema_fn=_mk_compute_schema_fn(MessageMeta),
                                         needed_columns=needed_columns)
    expected_needed_columns = needed_columns or collections.OrderedDict()
    assert wrapped_stage._needed_columns == expected_needed_columns


@pytest.mark.parametrize("use_accept_type_annotation", [True, False])
@pytest.mark.parametrize("accept_type, return_type",
                         [(pd.DataFrame, MessageMeta), (int, int), (MessageMeta, MessageMeta), (typing.Any, bool),
                          (typing.Union[float, int], float), (float, typing.Any), (typing.Any, float),
                          (typing.Any, typing.Any)])
def test_stage_decorator(config: Config, accept_type: type, return_type: type, use_accept_type_annotation: bool):

    if use_accept_type_annotation:

        @stage
        def test_fn(message: accept_type) -> return_type:
            return message
    else:

        @stage(accept_type=accept_type)
        def test_fn(message) -> return_type:
            return message

    wrapped_stage = test_fn(config)

    assert isinstance(wrapped_stage, WrappedFunctionStage)
    assert wrapped_stage.accepted_types() == (accept_type, )


@pytest.mark.parametrize("name", [None, "unittest-stage"])
def test_stage_decorator_name(config: Config, name: str):
    if name is None:
        expected_name = 'test_fn'
    else:
        expected_name = name

    @stage(name=name)
    def test_fn(message: float, value: float) -> float:
        return message * value

    wrapped_stage = test_fn(config, value=2.2)
    assert wrapped_stage.name == expected_name


@pytest.mark.parametrize("explicit_compute_schema_fn", [True, False])
@pytest.mark.parametrize("accept_type, return_type",
                         [(pd.DataFrame, MessageMeta), (int, int), (MessageMeta, MessageMeta), (typing.Any, bool),
                          (typing.Union[float, int], float), (float, float), (typing.Any, float),
                          (typing.Any, typing.Any)])
def test_stage_decorator_output_types(config: Config,
                                      accept_type: type,
                                      return_type: type,
                                      explicit_compute_schema_fn: bool):
    # For non-source types we need an upstream before we can check the compute_schema method outside of a pipeline
    @source
    def source_fn() -> accept_type:
        yield None

    if explicit_compute_schema_fn:
        mock_compute_schema_fn = mock.MagicMock()
        mock_compute_schema_fn.side_effect = _mk_compute_schema_fn(return_type)

        @stage(compute_schema_fn=mock_compute_schema_fn)
        def test_stage(message: accept_type) -> return_type:
            return message
    else:

        @stage
        def test_stage(message: accept_type) -> return_type:
            return message

    pipe = LinearPipeline(config)
    pipe.set_source(source_fn(config))  # pylint: disable=too-many-function-args
    wrapped_stage = pipe.add_stage(test_stage(config))
    pipe.build()

    if explicit_compute_schema_fn:
        mock_compute_schema_fn.assert_called_once()  # pipe.build() will call wrapped_stage.compute_schema()

    schema = StageSchema(wrapped_stage)
    wrapped_stage.compute_schema(schema)
    assert schema.output_schema.get_type() is return_type


def test_stage_decorator_no_annotation_error(config: Config):

    @stage
    def test_fn(message):
        return message

    with pytest.raises(ValueError):
        test_fn(config)


def test_stage_arg_no_value_error(config: Config):

    @stage
    def test_fn(message: float, value: float) -> float:
        return message * value

    with pytest.raises(ValueError):
        test_fn(config)  # pylint: disable=no-value-for-parameter


@pytest.mark.parametrize("needed_columns",
                         [None, {
                             'result': TypeId.INT64
                         }, {
                             'a': TypeId.INT64, 'b': TypeId.FLOAT32, 'c': TypeId.STRING
                         }])
def test_stage_decorator_needed_columns(config: Config, needed_columns: dict[str, TypeId]):

    @stage(needed_columns=needed_columns)
    def test_fn(message: MessageMeta) -> MessageMeta:
        return message

    wrapped_stage = test_fn(config)
    expected_needed_columns = needed_columns or collections.OrderedDict()
    assert wrapped_stage._needed_columns == expected_needed_columns


@pytest.mark.gpu_and_cpu_mode
def test_end_to_end_pipe(config: Config, filter_probs_df: DataFrameType):

    @source(execution_modes=(ExecutionMode.GPU, ExecutionMode.CPU))
    def source_gen(*, dataframes: list[DataFrameType]) -> collections.abc.Iterator[MessageMeta]:
        for df in dataframes:
            yield MessageMeta(df)

    @stage(execution_modes=(ExecutionMode.GPU, ExecutionMode.CPU))
    def multiplier(message: MessageMeta, *, column: str, value: int | float = 2.0) -> MessageMeta:
        with message.mutable_dataframe() as df:
            df[column] = df[column] * value

        return message

    multipy_by = 5
    expected_df = filter_probs_df.copy(deep=True)
    expected_df['v2'] = expected_df['v2'] * multipy_by * 2.0

    pipe = LinearPipeline(config)
    pipe.set_source(source_gen(config, dataframes=[filter_probs_df]))  # pylint: disable=too-many-function-args
    pipe.add_stage(multiplier(config, column='v2', value=multipy_by))
    pipe.add_stage(multiplier(config, column='v2'))
    sink = pipe.add_stage(CompareDataFrameStage(config, expected_df))
    pipe.run()

    assert_results(sink.get_results())


@pytest.mark.parametrize("has_arg, type_hint, expected",
                         [(False, None, False), (True, float, False), (True, "int", False),
                          (True, mrc.Subscription, True), (True, Subscription, True), (True, "mrc.Subscription", True),
                          (True, "Subscription", True)])
def test_fn_receives_subscription(has_arg: bool, type_hint: typing.Any, expected: bool):
    if has_arg:

        def test_fn(first_arg: type_hint):  # pylint: disable=unused-argument
            pass
    else:

        def test_fn():
            pass

    sig = inspect.signature(test_fn)
    assert _fn_receives_subscription(sig) is expected
