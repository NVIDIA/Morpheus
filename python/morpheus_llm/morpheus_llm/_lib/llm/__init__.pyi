"""
        -----------------------
        .. currentmodule:: morpheus_llm.llm
        .. autosummary::
           :toctree: _generate

        """
from __future__ import annotations
import morpheus_llm._lib.llm
import typing
import morpheus._lib.messages
import mrc.core.coro
import mrc.core.segment

__all__ = [
    "InputMap",
    "LLMContext",
    "LLMEngine",
    "LLMEngineStage",
    "LLMLambdaNode",
    "LLMNode",
    "LLMNodeBase",
    "LLMNodeRunner",
    "LLMTask",
    "LLMTaskHandler"
]


class InputMap():
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, external_name: str, internal_name: str) -> None: ...
    @property
    def external_name(self) -> str:
        """
        The name of node that will be mapped to this input. Use a leading '/' to indicate it is a sibling node otherwise it will be treated as a parent node. Can also specify a specific node output such as '/sibling_node/output1' to map the output 'output1' of 'sibling_node' to this input. Can also use a wild card such as '/sibling_node/\*' to match all internal node names

        :type: str
        """
    @external_name.setter
    def external_name(self, arg0: str) -> None:
        """
        The name of node that will be mapped to this input. Use a leading '/' to indicate it is a sibling node otherwise it will be treated as a parent node. Can also specify a specific node output such as '/sibling_node/output1' to map the output 'output1' of 'sibling_node' to this input. Can also use a wild card such as '/sibling_node/\*' to match all internal node names
        """
    @property
    def internal_name(self) -> str:
        """
        The internal node name that the external node maps to. Must match an input returned from `get_input_names()` of the desired node. Defaults to '-' which is a placeholder for the default input of the node. Use a wildcard '\*' to match all inputs of the node (Must also use a wild card on the external mapping).

        :type: str
        """
    @internal_name.setter
    def internal_name(self, arg0: str) -> None:
        """
        The internal node name that the external node maps to. Must match an input returned from `get_input_names()` of the desired node. Defaults to '-' which is a placeholder for the default input of the node. Use a wildcard '\*' to match all inputs of the node (Must also use a wild card on the external mapping).
        """
    pass
class LLMContext():
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, prent: LLMContext, name: str, inputs: typing.List[InputMap]) -> None: ...
    @typing.overload
    def __init__(self, task: LLMTask, message: morpheus._lib.messages.ControlMessage) -> None: ...
    @typing.overload
    def get_input(self) -> object | None: ...
    @typing.overload
    def get_input(self, node_name: str) -> object | None: ...
    def get_inputs(self) -> object | None: ...
    def message(self) -> morpheus._lib.messages.ControlMessage: ...
    def push(self, name: str, inputs: typing.List[InputMap]) -> LLMContext: ...
    @typing.overload
    def set_output(self, output_name: str, output: object | None) -> None: ...
    @typing.overload
    def set_output(self, outputs: object | None) -> None: ...
    def task(self) -> LLMTask: ...
    @property
    def full_name(self) -> str:
        """
        :type: str
        """
    @property
    def input_map(self) -> typing.List[InputMap]:
        """
        :type: typing.List[InputMap]
        """
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def parent(self) -> LLMContext:
        """
        :type: LLMContext
        """
    @property
    def view_outputs(self) -> object | None:
        """
        :type: object | None
        """
    pass
class LLMNodeBase():
    def __init__(self) -> None: ...
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
    def get_input_names(self) -> typing.List[str]: 
        """
        Get the input names for the node.

        Returns
        -------
        list[str]
            The input names for the node
        """
    pass
class LLMEngineStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, engine: LLMEngine) -> None: ...
    pass
class LLMLambdaNode(LLMNodeBase):
    def __init__(self, fn: function) -> None: ...
    def execute(self, context: LLMContext) -> typing.Awaitable[LLMContext]: ...
    def get_input_names(self) -> typing.List[str]: ...
    pass
class LLMNode(LLMNodeBase):
    def __init__(self) -> None: ...
    def add_node(self, name: str, *, inputs: object = None, node: LLMNodeBase, is_output: bool = False) -> LLMNodeRunner: 
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
    pass
class LLMEngine(LLMNode, LLMNodeBase):
    def __init__(self) -> None: ...
    def add_task_handler(self, inputs: typing.List[typing.Union[InputMap, str, typing.Tuple[str, str], LLMNodeRunner]], handler: LLMTaskHandler) -> None: ...
    def run(self, message: morpheus._lib.messages.ControlMessage) -> typing.Awaitable[typing.List[morpheus._lib.messages.ControlMessage]]: ...
    pass
class LLMNodeRunner():
    def execute(self, context: LLMContext) -> typing.Awaitable[LLMContext]: ...
    @property
    def inputs(self) -> typing.List[InputMap]:
        """
        :type: typing.List[InputMap]
        """
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def parent_input_names(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    @property
    def sibling_input_names(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    pass
class LLMTask():
    def __getitem__(self, key: str) -> object: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, task_type: str, task_dict: dict) -> None: ...
    def __len__(self) -> int: ...
    def __setitem__(self, key: str, value: object) -> None: ...
    @typing.overload
    def get(self, key: str) -> object: ...
    @typing.overload
    def get(self, key: str, default_value: object) -> object: ...
    @property
    def task_type(self) -> str:
        """
        :type: str
        """
    pass
class LLMTaskHandler():
    """
    Acts as a sink for an `LLMEngine`, emitting results as a `ControlMessage`
    """
    def __init__(self) -> None: ...
    def get_input_names(self) -> typing.List[str]: 
        """
        Get the input names for the task handler.

        Returns
        -------
        list[str]
            The input names for the task handler.
        """
    def try_handle(self, context: LLMContext) -> typing.Awaitable[typing.Optional[typing.List[morpheus._lib.messages.ControlMessage]]]: 
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
    pass
__version__ = '25.2.0'
