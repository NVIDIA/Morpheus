"""

        -----------------------
        .. currentmodule:: morpheus_llm.llm
        .. autosummary::
           :toctree: _generate

        
"""
from __future__ import annotations
import morpheus._lib.messages
import mrc.core.segment
import typing
__all__ = ['InputMap', 'LLMContext', 'LLMEngine', 'LLMEngineStage', 'LLMLambdaNode', 'LLMNode', 'LLMNodeBase', 'LLMNodeRunner', 'LLMTask', 'LLMTaskHandler']
class InputMap:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, external_name: str, internal_name: str) -> None:
        ...
    @property
    def external_name(self) -> str:
        """
        The name of node that will be mapped to this input. Use a leading '/' to indicate it is a sibling node otherwise it will be treated as a parent node. Can also specify a specific node output such as '/sibling_node/output1' to map the output 'output1' of 'sibling_node' to this input. Can also use a wild card such as '/sibling_node/\*' to match all internal node names
        """
    @external_name.setter
    def external_name(self, arg0: str) -> None:
        ...
    @property
    def internal_name(self) -> str:
        """
        The internal node name that the external node maps to. Must match an input returned from `get_input_names()` of the desired node. Defaults to '-' which is a placeholder for the default input of the node. Use a wildcard '\*' to match all inputs of the node (Must also use a wild card on the external mapping).
        """
    @internal_name.setter
    def internal_name(self, arg0: str) -> None:
        ...
class LLMContext:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, prent: LLMContext, name: str, inputs: list[InputMap]) -> None:
        ...
    @typing.overload
    def __init__(self, task: LLMTask, message: morpheus._lib.messages.ControlMessage) -> None:
        ...
    @typing.overload
    def get_input(self) -> typing.Any | None:
        ...
    @typing.overload
    def get_input(self, node_name: str) -> typing.Any | None:
        ...
    def get_inputs(self) -> typing.Any | None:
        ...
    def message(self) -> morpheus._lib.messages.ControlMessage:
        ...
    def push(self, name: str, inputs: list[InputMap]) -> LLMContext:
        ...
    @typing.overload
    def set_output(self, outputs: typing.Any | None) -> None:
        ...
    @typing.overload
    def set_output(self, output_name: str, output: typing.Any | None) -> None:
        ...
    def task(self) -> LLMTask:
        ...
    @property
    def full_name(self) -> str:
        ...
    @property
    def input_map(self) -> list[InputMap]:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def parent(self) -> LLMContext:
        ...
    @property
    def view_outputs(self) -> typing.Any | None:
        ...
class LLMEngine(LLMNode):
    def __init__(self) -> None:
        ...
    def add_task_handler(self, inputs: list[InputMap | str | tuple[str, str] | LLMNodeRunner], handler: LLMTaskHandler) -> None:
        ...
    def run(self, message: morpheus._lib.messages.ControlMessage) -> typing.Awaitable[list[morpheus._lib.messages.ControlMessage]]:
        ...
class LLMEngineStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, engine: LLMEngine) -> None:
        ...
class LLMLambdaNode(LLMNodeBase):
    def __init__(self, fn: typing.Callable) -> None:
        ...
    def execute(self, context: LLMContext) -> typing.Awaitable[LLMContext]:
        ...
    def get_input_names(self) -> list[str]:
        ...
class LLMNode(LLMNodeBase):
    def __init__(self) -> None:
        ...
    def add_node(self, name: str, *, inputs: typing.Any = None, node: LLMNodeBase, is_output: bool = False) -> LLMNodeRunner:
        """
                        Add an LLMNode to the current node.
        
                        Parameters
                        ----------
                        name : str
                            The name of the node to add
        
                        inputs : list[tuple[str, str]], optional
                            List of input mappings to use for the node, in the form of `[(external_name, internal_name), ...]`
                            If unspecified the node's input_names will be used.
        
                        node : LLMNodeBase
                            The node to add
        
                        is_output : bool, optional
                            Indicates if the node is an output node, by default False
        """
class LLMNodeBase:
    def __init__(self) -> None:
        ...
    def execute(self, context: LLMContext) -> typing.Awaitable[LLMContext]:
        """
                        Execute the current node with the given `context` instance.
        
                        All inputs for the given node should be fetched from the context, typically by calling either
                        `context.get_inputs` to fetch all inputs as a `dict`, or `context.get_input` to fetch a specific input.
        
                        Similarly the output of the node is written to the context using `context.set_output`.
        
                        Parameters
                        ----------
                        context : `morpheus._lib.llm.LLMContext`
                            Context instance to use for the execution
        """
    def get_input_names(self) -> list[str]:
        """
                        Get the input names for the node.
        
                        Returns
                        -------
                        list[str]
                            The input names for the node
        """
class LLMNodeRunner:
    def execute(self, context: LLMContext) -> typing.Awaitable[LLMContext]:
        ...
    @property
    def inputs(self) -> list[InputMap]:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def parent_input_names(self) -> list[str]:
        ...
    @property
    def sibling_input_names(self) -> list[str]:
        ...
class LLMTask:
    def __getitem__(self, key: str) -> typing.Any:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, task_type: str, task_dict: dict) -> None:
        ...
    def __len__(self) -> int:
        ...
    def __setitem__(self, key: str, value: typing.Any) -> None:
        ...
    @typing.overload
    def get(self, key: str) -> typing.Any:
        ...
    @typing.overload
    def get(self, key: str, default_value: typing.Any) -> typing.Any:
        ...
    @property
    def task_type(self) -> str:
        ...
class LLMTaskHandler:
    """
    Acts as a sink for an `LLMEngine`, emitting results as a `ControlMessage`
    """
    def __init__(self) -> None:
        ...
    def get_input_names(self) -> list[str]:
        """
                        Get the input names for the task handler.
        
                        Returns
                        -------
                        list[str]
                            The input names for the task handler.
        """
    def try_handle(self, context: LLMContext) -> typing.Awaitable[list[morpheus._lib.messages.ControlMessage] | None]:
        """
                        Convert the given `context` into a list of `ControlMessage` instances.
        
                        Parameters
                        ----------
                        context : `morpheus._lib.llm.LLMContext`
                            Context instance to use for the execution
        
                        Returns
                        -------
                        Task[Optional[list[ControlMessage]]]
        """
__version__: str = '25.2.0'
