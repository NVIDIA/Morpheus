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

import functools
import inspect
import typing
from abc import abstractmethod

import mrc
import mrc.core.operators as ops

from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.utils.shared_process_pool import SharedProcessPool

InputT = typing.TypeVar('InputT')
OutputT = typing.TypeVar('OutputT')


class MultiProcessingBaseStage(SinglePortStage, typing.Generic[InputT, OutputT]):
    """
    Base class for all MultiProcessing stages that make use of the SharedProcessPool.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    process_pool_usage : float
        The fraction of the process pool workers that this stage could use. Should be between 0 and 1.
    max_in_flight_messages : int, default = None
        The number of progress engines used by the stage. If None, it will be set to 1.5 times the total
        number of process pool workers

    Raises
    ------
    ValueError
        If the process pool usage is not between 0 and 1.
    """

    def __init__(self, *, c: Config, process_pool_usage: float, max_in_flight_messages: int = None):

        super().__init__(c=c)

        if not 0 <= process_pool_usage <= 1:
            raise ValueError("process_pool_usage must be between 0 and 1.")
        self._process_pool_usage = process_pool_usage

        self._shared_process_pool = SharedProcessPool()
        self._shared_process_pool.wait_until_ready()

        if max_in_flight_messages is None:
            # set the multiplier to 1.5 to keep the workers busy
            self._max_in_flight_messages = int(self._shared_process_pool.total_max_workers * 1.5)
        else:
            self._max_in_flight_messages = max_in_flight_messages

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Raises
        ------
        RuntimeError
            If the accepted types cannot be deduced from either __orig_class__ or __orig_bases__.

        Returns
        -------
        typing.Tuple
            Accepted input types.
        """

        # There are two approaches to inherit from this class:
        #     - With generic types: MultiProcessingDerivedStage(MultiProcessingBaseStage[InputT, OutputT])
        #     - With concrete types: MultiProcessingDerivedStage(MultiProcessingBaseStage[int, str])

        # When inheriting with generic types, the derived class can be instantiated like this:

        #     stage = MultiProcessingDerivedStage[int, str]()

        # In this case, typing.Generic stores the stage type in stage.__orig_class__, the concrete types can be accessed
        # as below:

        #     input_type = typing.get_args(stage.__orig_class__)[0] # int
        #     output_type = typing.get_args(stage.__orig_class__)[1] # str

        # However, when instantiating a stage which inherits with concrete types:

        #     stage = MultiProcessingDerivedStage()

        # The stage instance does not have __orig_class__ attribute (since it is not a generic type). Thus, the concrete
        # types need be retrieved from its base class (which is a generic type):

        #     input_type = typing.get_args(stage.__orig_bases__[0])[0] # int
        #     output_type = typing.get_args(stage.__orig_bases__[0])[1] # str

        if hasattr(self, "__orig_class__"):
            # inherited with generic types
            input_type = typing.get_args(self.__orig_class__)[0]  # pylint: disable=no-member

        elif hasattr(self, "__orig_bases__"):
            # inherited with concrete types
            input_type = typing.get_args(self.__orig_bases__[0])[0]  # pylint: disable=no-member

        else:
            raise RuntimeError("Could not deduct input type")

        return (input_type, )

    def compute_schema(self, schema: StageSchema):
        """
        Compute the output schema for the stage.

        Parameters
        ----------
        schema : StageSchema
            The schema for the stage.

        Raises
        ------
        RuntimeError
            If the output type cannot be deduced from either __orig_class__ or __orig_bases__.
        """

        # See the comment on `accepted_types` for more information on accessing the input and output types.
        if hasattr(self, "__orig_class__"):
            # inherited with abstract types
            output_type = typing.get_args(self.__orig_class__)[1]  # pylint: disable=no-member

        elif hasattr(self, "__orig_bases__"):
            # inherited with concrete types
            output_type = typing.get_args(self.__orig_bases__[0])[1]

        else:
            raise RuntimeError("Could not deduct output type")

        schema.output_schema.set_type(output_type)

    def supports_cpp_node(self):
        """Whether this stage supports a C++ node."""
        return False

    @abstractmethod
    def _on_data(self, data: InputT) -> OutputT:
        pass

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.name, ops.map(self._on_data))
        node.launch_options.pe_count = self._max_in_flight_messages

        builder.make_edge(input_node, node)

        return node


def _get_func_signature(func: typing.Callable[[InputT], OutputT]) -> tuple[type, type]:
    signature = inspect.signature(func)

    if isinstance(func, functools.partial):
        # If the function is a partial, find the type of the first unbound argument
        bound_args = func.keywords
        input_arg = None

        for param in signature.parameters.values():
            if param.name not in bound_args:
                if input_arg is not None:
                    raise ValueError("Found more than one unbound arguments in partial function")
                input_arg = param

        if input_arg is None:
            raise ValueError("Cannot find unbound argument in partial function")
        input_t = input_arg.annotation

    else:
        if len(signature.parameters) != 1:
            raise ValueError("Function must have exactly one argument")

        input_t = next(iter(signature.parameters.values())).annotation

    output_t = signature.return_annotation

    return (input_t, output_t)


class MultiProcessingStage(MultiProcessingBaseStage[InputT, OutputT]):
    """
    A derived class of MultiProcessingBaseStage that allows the user to define a process function that will be executed
    based on shared process pool.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    unique_name : str
        A unique name for the stage.
    process_fn:  typing.Callable[[InputT], OutputT]
        The function that will be executed in the process pool.
    max_in_flight_messages : int, default = None
        The number of progress engines used by the stage.
    """

    def __init__(self,
                 *,
                 c: Config,
                 unique_name: str,
                 process_fn: typing.Callable[[InputT], OutputT],
                 process_pool_usage: float,
                 max_in_flight_messages: int = None):
        super().__init__(c=c, process_pool_usage=process_pool_usage, max_in_flight_messages=max_in_flight_messages)

        self._name = unique_name
        self._process_fn = process_fn
        self._shared_process_pool.set_usage(self.name, self._process_pool_usage)

    @property
    def name(self) -> str:
        """Return the name of the stage."""
        return self._name

    def supported_execution_modes(self) -> tuple[ExecutionMode]:
        """
        Returns a tuple of supported execution modes of this stage.
        """
        return (ExecutionMode.GPU, ExecutionMode.CPU)

    def _on_data(self, data: InputT) -> OutputT:
        task = self._shared_process_pool.submit_task(self.name, self._process_fn, data)
        result = task.result()

        return result

    @staticmethod
    def create(*,
               c: Config,
               unique_name: str,
               process_fn: typing.Callable[[InputT], OutputT],
               process_pool_usage: float):
        """
        Create a MultiProcessingStage instance by deducing the input and output types from the process function.

        Parameters
        ----------
        c : morpheus.config.Config
            Pipeline configuration instance.
        unique_name : str
            A unique name for the stage.
        process_fn : typing.Callable[[InputT], OutputT]
            The function that will be executed in the process pool.
        process_pool_usage : float
            The fraction of the process pool workers that this stage could use. Should be between 0 and 1.

        Returns
        -------
        MultiProcessingStage[InputT, OutputT]
            A MultiProcessingStage instance with deduced input and output types.
        """

        input_t, output_t = _get_func_signature(process_fn)
        return MultiProcessingStage[input_t, output_t](c=c,
                                                       unique_name=unique_name,
                                                       process_pool_usage=process_pool_usage,
                                                       process_fn=process_fn)
