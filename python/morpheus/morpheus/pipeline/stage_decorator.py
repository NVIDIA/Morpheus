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

import collections
import functools
import inspect
import logging
import typing

import mrc
import pandas as pd
from mrc.core import operators as ops

import morpheus.pipeline as _pipeline  # pylint: disable=cyclic-import
from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.messages import MessageMeta

logger = logging.getLogger(__name__)

_InputT = typing.TypeVar('_InputT')
_OutputT = typing.TypeVar('_OutputT')
_P = typing.ParamSpec('_P')

GeneratorType = typing.Callable[_P, collections.abc.Iterator[_OutputT]]
ComputeSchemaType = typing.Callable[[_pipeline.StageSchema], None]


def _get_name_from_fn(fn: typing.Callable) -> str:
    try:
        return fn.__name__
    except AttributeError:
        # If the function is a partial, it won't have a name
        if isinstance(fn, functools.partial):
            return _get_name_from_fn(fn.func)

        return str(fn)


def _validate_keyword_arguments(fn_name: str,
                                signature: inspect.Signature,
                                kwargs: dict[str, typing.Any],
                                param_iter: typing.Iterator):
    # If we have any keyword arguments with a default value that we did not receive an explicit value for, we need
    # to bind it, otherwise it will trigger an error when MRC.
    for param in param_iter:

        if param.default is not signature.empty and param.name not in kwargs:
            kwargs[param.name] = param.default

        # if a parameter is keyword only, containing neither a a default value or an entry in on_data_kwargs, we
        # need to raise an error
        if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY) and param.name not in kwargs:
            raise ValueError(f"Wrapped function {fn_name} has keyword only parameter '{param.name}' that was not "
                             "provided a value")

        if param.kind is param.POSITIONAL_ONLY:

            raise ValueError("Positional arguments are not supported for wrapped functions. "
                             f"{fn_name} contains '{param.name}' that was not provided with a value")


def _fn_receives_subscription(signature: inspect.Signature) -> bool:
    """
    Inspects a function's signature to determine if the function expects a `mrc.Subscription` as it's first argument.
    This requires the type-hint to be set to either `mrc.Subscription` or a string hint of `"mrc.Subscription"` or
    `"Subscription"`

    Parameters
    ----------
    signature : `inspect.Signature` | `typing.Callable`
        Function or signature of a function to be checked

    Returns
    -------
    bool
        True if the signature indicates the first argument to be a `mrc.Subscription`.
    """
    param_iter = iter(signature.parameters.values())

    try:
        first_param = next(param_iter)
        if (first_param.annotation is mrc.Subscription or first_param.annotation == "mrc.Subscription"
                or first_param.annotation == "Subscription"):
            return True
    except StopIteration:
        pass

    return False


class WrappedFunctionSourceStage(_pipeline.SingleOutputSource):
    """
    Source stage that wraps a generator function as the method for generating messages.

    The wrapped function must be a generator function.  If the output type of the generator is an instance of
    `pandas.DataFrame`, `cudf.DataFrame` or `MessageMeta` the `PreAllocatedWrappedFunctionStage` class
    should be used instead, as this will also perform any DataFrame column allocations needed by other stages in the
    pipeline.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration instance.
    name: `str`
        Name of the stage.
    gen_fn : `GeneratorType`
        Generator function to use as the source of messages.
    compute_schema_fn : `ComputeSchemaType`
        Function to use for computing the schema of the stage.
    """

    def __init__(self,
                 config: Config,
                 *,
                 name: str,
                 gen_fn: GeneratorType,
                 compute_schema_fn: ComputeSchemaType,
                 execution_modes: tuple[ExecutionMode] = (ExecutionMode.GPU, )):
        super().__init__(config)
        # collections.abc.Generator is a subclass of collections.abc.Iterator
        if not inspect.isgeneratorfunction(gen_fn):
            raise ValueError("Wrapped source functions must be generator functions")

        self._name = name
        self._gen_fn = gen_fn
        self._compute_schema_fn = compute_schema_fn
        self._supported_execution_modes = execution_modes

    @property
    def name(self) -> str:
        return self._name

    def supports_cpp_node(self) -> bool:
        return False

    def compute_schema(self, schema: _pipeline.StageSchema):
        self._compute_schema_fn(schema)

    def supported_execution_modes(self) -> tuple[ExecutionMode]:
        """
        Returns a tuple of supported execution modes of this stage.
        """
        return self._supported_execution_modes

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:
        return builder.make_source(self.unique_name, self._gen_fn)


class PreAllocatedWrappedFunctionStage(_pipeline.PreallocatorMixin, WrappedFunctionSourceStage):
    """
    Source stage that wraps a generator function as the method for generating messages.

    This stage provides the same functionality as `WrappedFunctionSourceStage`, but also performs any DataFrame column
    allocations needed by other stages in the pipeline. As such the return type for `gen_fn` must be one of:
    `pandas.DataFrame`, `cudf.DataFrame`, or `MessageMeta`.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration instance.
    name: `str`
        Name of the stage.
    gen_fn : `GeneratorType`
        Generator function to use as the source of messages.
    compute_schema_fn : `ComputeSchemaType`
        Function to use for computing the schema of the stage.
    """


def source(
    gen_fn: GeneratorType = None,
    *,
    name: str = None,
    compute_schema_fn: ComputeSchemaType = None,
    execution_modes: tuple[ExecutionMode] = (ExecutionMode.GPU, )
) -> typing.Callable[typing.Concatenate[Config, _P], WrappedFunctionSourceStage]:
    """
    Decorator for wrapping a function as a source stage. The function must be a generator method, and provide a
    provide a return type annotation.

    When `compute_schema_fn` is `None`, the return type annotation will be used by the stage as the output type.

    When invoked the wrapped function will return a source stage, any additional keyword arguments passed in aside from
    the config, will be bound to the wrapped function via `functools.partial`.

    Examples
    --------

    >>> @source
    ... def source_gen(*, dataframes: list[cudf.DataFrame]) -> collections.abc.Iterator[MessageMeta]:
    ...     for df in dataframes:
    ...         yield MessageMeta(df)
    ...
    >>>

    >>> pipe.set_source(source_gen(config, dataframes=[df]))
    """
    if gen_fn is None:
        return functools.partial(source,
                                 name=name,
                                 compute_schema_fn=compute_schema_fn,
                                 execution_modes=execution_modes)

    # Use wraps to ensure user's don't lose their function name and docstrinsgs, however we do want to override the
    # annotations to reflect that the returned function requires a config and returns a stage
    @functools.wraps(gen_fn, assigned=('__module__', '__name__', '__qualname__', '__doc__'))
    def wrapper(config: Config, **kwargs: _P.kwargs) -> WrappedFunctionSourceStage:
        nonlocal name
        nonlocal compute_schema_fn

        if name is None:
            name = _get_name_from_fn(gen_fn)

        signature = inspect.signature(gen_fn)
        return_type = signature.return_annotation
        if return_type is signature.empty:
            raise ValueError("Source functions must specify a return type annotation")

        # Try and determine if the method expects a reference to a subscription object
        param_iter = iter(signature.parameters.values())
        if _fn_receives_subscription(signature):
            next(param_iter)  # advance the iterator past the subscription

        # We need to unpack generator and iterator return types to get the actual type of the yielded type.
        # When someone uses collections.abc.Generator or collections.abc.Iterator the return type is an instance of
        # typing.GenericAlias, however when someone uses typing.Generator or typing.Iterator the return type is an
        # instance of typing._GenericAlias. We need to check for both.
        if isinstance(return_type, (typing.GenericAlias, typing._GenericAlias)):
            return_type = return_type.__args__[0]

        if compute_schema_fn is None:

            def compute_schema_fn_inner(schema: _pipeline.StageSchema):
                schema.output_schema.set_type(return_type)

            compute_schema_fn = compute_schema_fn_inner

        _validate_keyword_arguments(name, signature, kwargs, param_iter=param_iter)

        bound_gen_fn = functools.partial(gen_fn, **kwargs)

        pre_allocation_output_types = [pd.DataFrame, MessageMeta]
        if config.execution_mode == ExecutionMode.GPU:
            import cudf
            pre_allocation_output_types.append(cudf.DataFrame)

        # If the return type supports pre-allocation we use the pre-allocating source
        if return_type in pre_allocation_output_types:

            return PreAllocatedWrappedFunctionStage(config=config,
                                                    name=name,
                                                    gen_fn=bound_gen_fn,
                                                    compute_schema_fn=compute_schema_fn,
                                                    execution_modes=execution_modes)

        return WrappedFunctionSourceStage(config=config,
                                          name=name,
                                          gen_fn=bound_gen_fn,
                                          compute_schema_fn=compute_schema_fn,
                                          execution_modes=execution_modes)

    return wrapper


class WrappedFunctionStage(_pipeline.SinglePortStage):
    """
    Stage that wraps a function to be used for processing messages.

    The function must receive at least one argument, the first argument must be the incoming message, and must
    return a value.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration instance.
    name: `str`
        Name of the stage.
    on_data_fn : `typing.Callable`
        Function to be used for processing messages.
    accept_type: type
        Type of the input message.
    compute_schema_fn : `ComputeSchemaType`
        Function to use for computing the schema of the stage.
    needed_columns : `dict[str, TypeId]`, optional
        Dictionary of column names and types that the function requires to be present in the DataFrame. This is used
        by the `PreAllocatedWrappedFunctionStage` to ensure the DataFrame has the needed columns allocated.
    """

    def __init__(self,
                 config: Config,
                 *,
                 name: str = None,
                 on_data_fn: typing.Callable,
                 accept_type: type,
                 compute_schema_fn: ComputeSchemaType,
                 needed_columns: dict[str, TypeId] = None,
                 execution_modes: tuple[ExecutionMode] = (ExecutionMode.GPU, )):
        super().__init__(config)
        self._name = name
        self._on_data_fn = on_data_fn
        self._accept_type = accept_type
        self._compute_schema_fn = compute_schema_fn

        if needed_columns is not None:
            self._needed_columns.update(needed_columns)

        self._supported_execution_modes = execution_modes

    @property
    def name(self) -> str:
        return self._name

    def accepted_types(self) -> typing.Tuple:
        return (self._accept_type, )

    def supports_cpp_node(self) -> bool:
        return False

    def compute_schema(self, schema: _pipeline.StageSchema):
        self._compute_schema_fn(schema)

    def supported_execution_modes(self) -> tuple[ExecutionMode]:
        """
        Returns a tuple of supported execution modes of this stage.
        """
        return self._supported_execution_modes

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self._on_data_fn))
        builder.make_edge(input_node, node)

        return node


DecoratedStageType = typing.Callable[typing.Concatenate[Config, _P], WrappedFunctionStage]


def stage(
    on_data_fn: typing.Callable[typing.Concatenate[_InputT, _P], _OutputT] = None,
    *,
    name: str = None,
    accept_type: type = None,
    compute_schema_fn: ComputeSchemaType = None,
    needed_columns: dict[str, TypeId] = None,
    execution_modes: tuple[ExecutionMode] = (ExecutionMode.GPU, )
) -> DecoratedStageType:
    """
    Decorator for wrapping a function as a stage. The function must receive at least one argument, the first argument
    must be the incoming message, and must return a value.

    It is required to use type annotations for the function parameters and return type, as this will be used
    by the stage as the accept and output types. If the incoming message parameter has no type annotation, the stage
    will be use `typing.Any` as the input type. If the return type has no type annotation, the stage will be set to
    return the same type as the input type.

    When invoked the wrapped function will return a stage, any additional arguments passed in aside from the config,
    will be bound to the wrapped function via `functools.partial`.

    Examples
    --------

    >>> @stage
    ... def multiplier(message: MessageMeta, *, column: str, value: int | float) -> MessageMeta:
    ...     with message.mutable_dataframe() as df:
    ...         df[column] = df[column] * value
    ...
    ...     return message
    ...
    >>>

    >>> pipe.add_stage(multiplier(config, column='v2', value=5))

    >>> # This will fail since `column` is required but no default value is provided:
    >>> pipe.add_stage(multiplier(config, value=5))
    """

    if on_data_fn is None:
        return functools.partial(stage,
                                 name=name,
                                 accept_type=accept_type,
                                 compute_schema_fn=compute_schema_fn,
                                 needed_columns=needed_columns,
                                 execution_modes=execution_modes)

    # Use wraps to ensure user's don't lose their function name and docstrinsgs, however we do want to override the
    # annotations to reflect that the returned function requires a config and returns a stage
    @functools.wraps(on_data_fn, assigned=('__module__', '__name__', '__qualname__', '__doc__'))
    def wrapper(config: Config, **kwargs: _P.kwargs) -> WrappedFunctionStage:
        nonlocal name
        nonlocal accept_type
        nonlocal compute_schema_fn

        if name is None:
            name = _get_name_from_fn(on_data_fn)

        # Even if both accept_type and compute_schema_fn are provided, we should still need to inspect the function
        # signature to verify it is callable with at least one argument
        signature = inspect.signature(on_data_fn)
        param_iter = iter(signature.parameters.values())

        try:
            first_param = next(param_iter)
            accept_type = accept_type or first_param.annotation
            if accept_type is signature.empty:
                raise ValueError(f"{first_param.name} argument of {name} has no type annotation")
        except StopIteration as e:
            raise ValueError(f"Stage function {name} must have at least one parameter") from e

        if compute_schema_fn is None:  # pylint: disable=used-before-assignment
            return_type = signature.return_annotation
            if return_type is signature.empty:
                raise ValueError(
                    "Stage functions must have either a return type annotation or specify a compute_schema_fn")

            def compute_schema_fn_inner(schema: _pipeline.StageSchema):
                if return_type is typing.Any:
                    out_type = schema.input_schema.get_type()
                else:
                    out_type = return_type

                schema.output_schema.set_type(out_type)

            compute_schema_fn = compute_schema_fn_inner

        _validate_keyword_arguments(name, signature, kwargs, param_iter=param_iter)

        bound_on_data_fn = functools.partial(on_data_fn, **kwargs)

        return WrappedFunctionStage(config=config,
                                    name=name,
                                    on_data_fn=bound_on_data_fn,
                                    accept_type=accept_type,
                                    compute_schema_fn=compute_schema_fn,
                                    needed_columns=needed_columns,
                                    execution_modes=execution_modes)

    return wrapper
