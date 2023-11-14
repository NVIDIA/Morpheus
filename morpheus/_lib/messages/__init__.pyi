"""
        -----------------------
        .. currentmodule:: morpheus.messages
        .. autosummary::
           :toctree: _generate

        """
from __future__ import annotations
import morpheus._lib.messages
import typing
import cupy
import morpheus._lib.common
import mrc.core.node

__all__ = [
    "ControlMessage",
    "ControlMessageType",
    "DataLoaderRegistry",
    "DataTable",
    "InferenceMemory",
    "InferenceMemoryFIL",
    "InferenceMemoryNLP",
    "MessageMeta",
    "MultiInferenceFILMessage",
    "MultiInferenceMessage",
    "MultiInferenceNLPMessage",
    "MultiMessage",
    "MultiResponseMessage",
    "MultiResponseProbsMessage",
    "MultiTensorMessage",
    "MutableTableCtxMgr",
    "ResponseMemory",
    "ResponseMemoryProbs",
    "TensorMemory",
    "cupy"
]


class ControlMessage():
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: ControlMessage) -> None: ...
    @typing.overload
    def __init__(self, arg0: dict) -> None: ...
    def add_task(self, task_type: str, task: dict) -> None: ...
    @typing.overload
    def config(self) -> dict: ...
    @typing.overload
    def config(self, config: dict) -> None: ...
    def copy(self) -> ControlMessage: ...
    def get_metadata(self, key: typing.Optional[str] = None) -> object: ...
    def get_tasks(self) -> dict: ...
    def has_metadata(self, key: str) -> bool: ...
    def has_task(self, task_type: str) -> bool: ...
    def list_metadata(self) -> dict: ...
    @typing.overload
    def payload(self) -> MessageMeta: ...
    @typing.overload
    def payload(self, arg0: MessageMeta) -> None: ...
    def remove_task(self, task_type: str) -> dict: ...
    def set_metadata(self, key: str, value: object) -> None: ...
    @typing.overload
    def task_type(self) -> ControlMessageType: ...
    @typing.overload
    def task_type(self, task_type: ControlMessageType) -> None: ...
    pass
class ControlMessageType():
    """
    Members:

      INFERENCE

      NONE

      TRAINING
    """
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    INFERENCE: morpheus._lib.messages.ControlMessageType # value = <ControlMessageType.INFERENCE: 1>
    NONE: morpheus._lib.messages.ControlMessageType # value = <ControlMessageType.INFERENCE: 1>
    TRAINING: morpheus._lib.messages.ControlMessageType # value = <ControlMessageType.TRAINING: 2>
    __members__: dict # value = {'INFERENCE': <ControlMessageType.INFERENCE: 1>, 'NONE': <ControlMessageType.INFERENCE: 1>, 'TRAINING': <ControlMessageType.TRAINING: 2>}
    pass
class DataLoaderRegistry():
    @staticmethod
    def contains(name: str) -> bool: ...
    @staticmethod
    def list() -> typing.List[str]: ...
    @staticmethod
    def register_loader(name: str, loader: typing.Callable[[ControlMessage, dict], ControlMessage], throw_if_exists: bool = True) -> None: ...
    @staticmethod
    def unregister_loader(name: str, throw_if_not_exists: bool = True) -> None: ...
    pass
class DataTable():
    pass
class TensorMemory():
    def __init__(self, *, count: int, tensors: object = None) -> None: ...
    def get_tensor(self, name: str) -> object: ...
    def get_tensors(self) -> typing.Dict[str, object]: ...
    def has_tensor(self, arg0: str) -> bool: ...
    def set_tensor(self, name: str, tensor: object) -> None: ...
    def set_tensors(self, tensors: typing.Dict[str, object]) -> None: ...
    @property
    def count(self) -> int:
        """
        :type: int
        """
    @property
    def tensor_names(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    pass
class InferenceMemory(TensorMemory):
    def __init__(self, *, count: int, tensors: object = None) -> None: ...
    def get_input(self, name: str) -> object: ...
    def set_input(self, name: str, tensor: object) -> None: ...
    pass
class InferenceMemoryNLP(InferenceMemory, TensorMemory):
    def __init__(self, *, count: int, input_ids: object, input_mask: object, seq_ids: object) -> None: ...
    @property
    def input_ids(self) -> object:
        """
        :type: object
        """
    @input_ids.setter
    def input_ids(self, arg1: object) -> None:
        pass
    @property
    def input_mask(self) -> object:
        """
        :type: object
        """
    @input_mask.setter
    def input_mask(self, arg1: object) -> None:
        pass
    @property
    def seq_ids(self) -> object:
        """
        :type: object
        """
    @seq_ids.setter
    def seq_ids(self, arg1: object) -> None:
        pass
    pass
class MessageMeta():
    def __init__(self, df: object) -> None: ...
    def copy_dataframe(self) -> object: ...
    def ensure_sliceable_index(self) -> typing.Optional[str]: ...
    def has_sliceable_index(self) -> bool: ...
    @staticmethod
    def make_from_file(arg0: str) -> MessageMeta: ...
    def mutable_dataframe(self) -> MutableTableCtxMgr: ...
    @property
    def count(self) -> int:
        """
        :type: int
        """
    @property
    def df(self) -> object:
        """
        :type: object
        """
    pass
class MultiMessage():
    def __getstate__(self) -> dict: ...
    def __init__(self, *, meta: MessageMeta, mess_offset: int = 0, mess_count: int = -1) -> None: ...
    def __setstate__(self, arg0: dict) -> None: ...
    def copy_ranges(self, ranges: typing.List[typing.Tuple[int, int]], num_selected_rows: object = None) -> MultiMessage: ...
    @typing.overload
    def get_meta(self) -> object: ...
    @typing.overload
    def get_meta(self, arg0: None) -> object: ...
    @typing.overload
    def get_meta(self, arg0: str) -> object: ...
    @typing.overload
    def get_meta(self, arg0: typing.List[str]) -> object: ...
    def get_meta_list(self, arg0: object) -> object: ...
    def get_slice(self, arg0: int, arg1: int) -> MultiMessage: ...
    def set_meta(self, arg0: object, arg1: object) -> None: ...
    @property
    def mess_count(self) -> int:
        """
        :type: int
        """
    @property
    def mess_offset(self) -> int:
        """
        :type: int
        """
    @property
    def meta(self) -> MessageMeta:
        """
        :type: MessageMeta
        """
    pass
class MultiTensorMessage(MultiMessage):
    def __init__(self, *, meta: MessageMeta, mess_offset: int = 0, mess_count: int = -1, memory: TensorMemory, offset: int = 0, count: int = -1, id_tensor_name: str = 'seq_ids') -> None: ...
    def get_id_tensor(self) -> object: ...
    def get_tensor(self, arg0: str) -> object: ...
    @property
    def count(self) -> int:
        """
        :type: int
        """
    @property
    def memory(self) -> TensorMemory:
        """
        :type: TensorMemory
        """
    @property
    def offset(self) -> int:
        """
        :type: int
        """
    pass
class MultiInferenceMessage(MultiTensorMessage, MultiMessage):
    def __init__(self, *, meta: MessageMeta, mess_offset: int = 0, mess_count: int = -1, memory: TensorMemory, offset: int = 0, count: int = -1, id_tensor_name: str = 'seq_ids') -> None: ...
    def get_input(self, arg0: str) -> object: ...
    pass
class MultiInferenceFILMessage(MultiInferenceMessage, MultiTensorMessage, MultiMessage):
    def __init__(self, *, meta: MessageMeta, mess_offset: int = 0, mess_count: int = -1, memory: TensorMemory, offset: int = 0, count: int = -1, id_tensor_name: str = 'seq_ids') -> None: ...
    @property
    def input__0(self) -> object:
        """
        :type: object
        """
    @property
    def seq_ids(self) -> object:
        """
        :type: object
        """
    pass
class MultiResponseMessage(MultiTensorMessage, MultiMessage):
    def __init__(self, *, meta: MessageMeta, mess_offset: int = 0, mess_count: int = -1, memory: TensorMemory, offset: int = 0, count: int = -1, id_tensor_name: str = 'seq_ids', probs_tensor_name: str = 'probs') -> None: ...
    def get_output(self, arg0: str) -> object: ...
    def get_probs_tensor(self) -> object: ...
    @property
    def probs_tensor_name(self) -> str:
        """
        :type: str
        """
    @probs_tensor_name.setter
    def probs_tensor_name(self, arg1: str) -> None:
        pass
    pass
class MultiResponseProbsMessage(MultiResponseMessage, MultiTensorMessage, MultiMessage):
    def __init__(self, *, meta: MessageMeta, mess_offset: int = 0, mess_count: int = -1, memory: TensorMemory, offset: int = 0, count: int = -1, id_tensor_name: str = 'seq_ids', probs_tensor_name: str = 'probs') -> None: ...
    @property
    def probs(self) -> object:
        """
        :type: object
        """
    pass
class MultiInferenceNLPMessage(MultiInferenceMessage, MultiTensorMessage, MultiMessage):
    def __init__(self, *, meta: MessageMeta, mess_offset: int = 0, mess_count: int = -1, memory: TensorMemory, offset: int = 0, count: int = -1, id_tensor_name: str = 'seq_ids') -> None: ...
    @property
    def input_ids(self) -> object:
        """
        :type: object
        """
    @property
    def input_mask(self) -> object:
        """
        :type: object
        """
    @property
    def seq_ids(self) -> object:
        """
        :type: object
        """
    pass
class MutableTableCtxMgr():
    def __enter__(self) -> object: ...
    def __exit__(self, arg0: object, arg1: object, arg2: object) -> None: ...
    def __getattr__(self, *args, **kwargs) -> None: ...
    def __getitem__(self, *args, **kwargs) -> None: ...
    def __setattr__(self, *args, **kwargs) -> None: ...
    def __setitem__(self, *args, **kwargs) -> None: ...
    pass
class ResponseMemory(TensorMemory):
    def __init__(self, *, count: int, tensors: object = None) -> None: ...
    def get_output(self, name: str) -> object: ...
    def set_output(self, name: str, tensor: object) -> None: ...
    pass
class ResponseMemoryProbs(ResponseMemory, TensorMemory):
    def __init__(self, *, count: int, probs: object) -> None: ...
    @property
    def probs(self) -> object:
        """
        :type: object
        """
    @probs.setter
    def probs(self, arg1: object) -> None:
        pass
    pass
class InferenceMemoryFIL(InferenceMemory, TensorMemory):
    def __init__(self, *, count: int, input__0: object, seq_ids: object) -> None: ...
    @property
    def input__0(self) -> object:
        """
        :type: object
        """
    @input__0.setter
    def input__0(self, arg1: object) -> None:
        pass
    @property
    def seq_ids(self) -> object:
        """
        :type: object
        """
    @seq_ids.setter
    def seq_ids(self, arg1: object) -> None:
        pass
    pass
__version__ = '23.11.0'
