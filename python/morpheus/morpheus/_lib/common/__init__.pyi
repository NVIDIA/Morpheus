"""

        -----------------------
        .. currentmodule:: morpheus._lib.common
        .. autosummary::
           :toctree: _generate
        
"""
from __future__ import annotations
import os
import typing
__all__ = ['FiberQueue', 'FileTypes', 'FilterSource', 'HttpEndpoint', 'HttpServer', 'IndicatorsFontStyle', 'IndicatorsTextColor', 'Tensor', 'TypeId', 'determine_file_type', 'read_file_to_df', 'typeid_is_fully_supported', 'typeid_to_numpy_str', 'write_df_to_file']
class FiberQueue:
    def __enter__(self) -> FiberQueue:
        ...
    def __exit__(self, arg0: typing.Any, arg1: typing.Any, arg2: typing.Any) -> None:
        ...
    def __init__(self, max_size: int) -> None:
        ...
    def close(self) -> None:
        ...
    def get(self, block: bool = True, timeout: float = 0.0) -> typing.Any:
        ...
    def is_closed(self) -> bool:
        ...
    def put(self, item: typing.Any, block: bool = True, timeout: float = 0.0) -> None:
        ...
class FileTypes:
    """
    The type of files that the `FileSourceStage` can read and `WriteToFileStage` can write. Use 'auto' to determine from the file extension.
    
    Members:
    
      Auto
    
      JSON
    
      CSV
    
      PARQUET
    """
    Auto: typing.ClassVar[FileTypes]  # value = <FileTypes.Auto: 0>
    CSV: typing.ClassVar[FileTypes]  # value = <FileTypes.CSV: 2>
    JSON: typing.ClassVar[FileTypes]  # value = <FileTypes.JSON: 1>
    PARQUET: typing.ClassVar[FileTypes]  # value = <FileTypes.PARQUET: 3>
    __members__: typing.ClassVar[dict[str, FileTypes]]  # value = {'Auto': <FileTypes.Auto: 0>, 'JSON': <FileTypes.JSON: 1>, 'CSV': <FileTypes.CSV: 2>, 'PARQUET': <FileTypes.PARQUET: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class FilterSource:
    """
    Enum to indicate which source the FilterDetectionsStage should operate on.
    
    Members:
    
      Auto
    
      TENSOR
    
      DATAFRAME
    """
    Auto: typing.ClassVar[FilterSource]  # value = <FilterSource.Auto: 0>
    DATAFRAME: typing.ClassVar[FilterSource]  # value = <FilterSource.DATAFRAME: 2>
    TENSOR: typing.ClassVar[FilterSource]  # value = <FilterSource.TENSOR: 1>
    __members__: typing.ClassVar[dict[str, FilterSource]]  # value = {'Auto': <FilterSource.Auto: 0>, 'TENSOR': <FilterSource.TENSOR: 1>, 'DATAFRAME': <FilterSource.DATAFRAME: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class HttpEndpoint:
    def __init__(self, py_parse_fn: typing.Callable, url: str, method: str, include_headers: bool = False) -> None:
        ...
class HttpServer:
    def __enter__(self) -> HttpServer:
        ...
    def __exit__(self, arg0: typing.Any, arg1: typing.Any, arg2: typing.Any) -> None:
        ...
    def __init__(self, endpoints: list[HttpEndpoint], bind_address: str = '127.0.0.1', port: int = 8080, num_threads: int = 1, max_payload_size: int = 10485760, request_timeout: int = 30) -> None:
        ...
    def is_running(self) -> bool:
        ...
    def start(self) -> None:
        ...
    def stop(self) -> None:
        ...
class IndicatorsFontStyle:
    """
    Members:
    
      bold
    
      dark
    
      italic
    
      underline
    
      blink
    
      reverse
    
      concealed
    
      crossed
    """
    __members__: typing.ClassVar[dict[str, IndicatorsFontStyle]]  # value = {'bold': <IndicatorsFontStyle.bold: 0>, 'dark': <IndicatorsFontStyle.dark: 1>, 'italic': <IndicatorsFontStyle.italic: 2>, 'underline': <IndicatorsFontStyle.underline: 3>, 'blink': <IndicatorsFontStyle.blink: 4>, 'reverse': <IndicatorsFontStyle.reverse: 5>, 'concealed': <IndicatorsFontStyle.concealed: 6>, 'crossed': <IndicatorsFontStyle.crossed: 7>}
    blink: typing.ClassVar[IndicatorsFontStyle]  # value = <IndicatorsFontStyle.blink: 4>
    bold: typing.ClassVar[IndicatorsFontStyle]  # value = <IndicatorsFontStyle.bold: 0>
    concealed: typing.ClassVar[IndicatorsFontStyle]  # value = <IndicatorsFontStyle.concealed: 6>
    crossed: typing.ClassVar[IndicatorsFontStyle]  # value = <IndicatorsFontStyle.crossed: 7>
    dark: typing.ClassVar[IndicatorsFontStyle]  # value = <IndicatorsFontStyle.dark: 1>
    italic: typing.ClassVar[IndicatorsFontStyle]  # value = <IndicatorsFontStyle.italic: 2>
    reverse: typing.ClassVar[IndicatorsFontStyle]  # value = <IndicatorsFontStyle.reverse: 5>
    underline: typing.ClassVar[IndicatorsFontStyle]  # value = <IndicatorsFontStyle.underline: 3>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class IndicatorsTextColor:
    """
    Members:
    
      grey
    
      red
    
      green
    
      yellow
    
      blue
    
      magenta
    
      cyan
    
      white
    
      unspecified
    """
    __members__: typing.ClassVar[dict[str, IndicatorsTextColor]]  # value = {'grey': <IndicatorsTextColor.grey: 0>, 'red': <IndicatorsTextColor.red: 1>, 'green': <IndicatorsTextColor.green: 2>, 'yellow': <IndicatorsTextColor.yellow: 3>, 'blue': <IndicatorsTextColor.blue: 4>, 'magenta': <IndicatorsTextColor.magenta: 5>, 'cyan': <IndicatorsTextColor.cyan: 6>, 'white': <IndicatorsTextColor.white: 7>, 'unspecified': <IndicatorsTextColor.unspecified: 8>}
    blue: typing.ClassVar[IndicatorsTextColor]  # value = <IndicatorsTextColor.blue: 4>
    cyan: typing.ClassVar[IndicatorsTextColor]  # value = <IndicatorsTextColor.cyan: 6>
    green: typing.ClassVar[IndicatorsTextColor]  # value = <IndicatorsTextColor.green: 2>
    grey: typing.ClassVar[IndicatorsTextColor]  # value = <IndicatorsTextColor.grey: 0>
    magenta: typing.ClassVar[IndicatorsTextColor]  # value = <IndicatorsTextColor.magenta: 5>
    red: typing.ClassVar[IndicatorsTextColor]  # value = <IndicatorsTextColor.red: 1>
    unspecified: typing.ClassVar[IndicatorsTextColor]  # value = <IndicatorsTextColor.unspecified: 8>
    white: typing.ClassVar[IndicatorsTextColor]  # value = <IndicatorsTextColor.white: 7>
    yellow: typing.ClassVar[IndicatorsTextColor]  # value = <IndicatorsTextColor.yellow: 3>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Tensor:
    @staticmethod
    def from_cupy(arg0: typing.Any) -> Tensor:
        ...
    def to_cupy(self) -> typing.Any:
        ...
    @property
    def __cuda_array_interface__(self) -> dict:
        ...
class TypeId:
    """
    Supported Morpheus types
    
    Members:
    
      EMPTY
    
      INT8
    
      INT16
    
      INT32
    
      INT64
    
      UINT8
    
      UINT16
    
      UINT32
    
      UINT64
    
      FLOAT32
    
      FLOAT64
    
      BOOL8
    
      STRING
    """
    BOOL8: typing.ClassVar[TypeId]  # value = <TypeId.BOOL8: 11>
    EMPTY: typing.ClassVar[TypeId]  # value = <TypeId.EMPTY: 0>
    FLOAT32: typing.ClassVar[TypeId]  # value = <TypeId.FLOAT32: 9>
    FLOAT64: typing.ClassVar[TypeId]  # value = <TypeId.FLOAT64: 10>
    INT16: typing.ClassVar[TypeId]  # value = <TypeId.INT16: 2>
    INT32: typing.ClassVar[TypeId]  # value = <TypeId.INT32: 3>
    INT64: typing.ClassVar[TypeId]  # value = <TypeId.INT64: 4>
    INT8: typing.ClassVar[TypeId]  # value = <TypeId.INT8: 1>
    STRING: typing.ClassVar[TypeId]  # value = <TypeId.STRING: 12>
    UINT16: typing.ClassVar[TypeId]  # value = <TypeId.UINT16: 6>
    UINT32: typing.ClassVar[TypeId]  # value = <TypeId.UINT32: 7>
    UINT64: typing.ClassVar[TypeId]  # value = <TypeId.UINT64: 8>
    UINT8: typing.ClassVar[TypeId]  # value = <TypeId.UINT8: 5>
    __members__: typing.ClassVar[dict[str, TypeId]]  # value = {'EMPTY': <TypeId.EMPTY: 0>, 'INT8': <TypeId.INT8: 1>, 'INT16': <TypeId.INT16: 2>, 'INT32': <TypeId.INT32: 3>, 'INT64': <TypeId.INT64: 4>, 'UINT8': <TypeId.UINT8: 5>, 'UINT16': <TypeId.UINT16: 6>, 'UINT32': <TypeId.UINT32: 7>, 'UINT64': <TypeId.UINT64: 8>, 'FLOAT32': <TypeId.FLOAT32: 9>, 'FLOAT64': <TypeId.FLOAT64: 10>, 'BOOL8': <TypeId.BOOL8: 11>, 'STRING': <TypeId.STRING: 12>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
@typing.overload
def determine_file_type(filename: str) -> FileTypes:
    ...
@typing.overload
def determine_file_type(filename: os.PathLike) -> FileTypes:
    ...
def read_file_to_df(filename: str, file_type: FileTypes = FileTypes.Auto) -> typing.Any:
    ...
def typeid_is_fully_supported(arg0: TypeId) -> bool:
    ...
def typeid_to_numpy_str(arg0: TypeId) -> str:
    ...
def write_df_to_file(df: typing.Any, filename: str, file_type: FileTypes = FileTypes.Auto, **kwargs) -> None:
    ...
__version__: str = '25.2.0'
