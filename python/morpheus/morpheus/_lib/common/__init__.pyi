"""
        -----------------------
        .. currentmodule:: morpheus.common
        .. autosummary::
           :toctree: _generate
        """
from __future__ import annotations
import morpheus._lib.common
import typing
import os

__all__ = [
    "FiberQueue",
    "FileTypes",
    "FilterSource",
    "HttpEndpoint",
    "HttpServer",
    "IndicatorsFontStyle",
    "IndicatorsTextColor",
    "Tensor",
    "TypeId",
    "determine_file_type",
    "read_file_to_df",
    "typeid_is_fully_supported",
    "typeid_to_numpy_str",
    "write_df_to_file"
]


class FiberQueue():
    def __enter__(self) -> FiberQueue: ...
    def __exit__(self, arg0: object, arg1: object, arg2: object) -> None: ...
    def __init__(self, max_size: int) -> None: ...
    def close(self) -> None: ...
    def get(self, block: bool = True, timeout: float = 0.0) -> object: ...
    def is_closed(self) -> bool: ...
    def put(self, item: object, block: bool = True, timeout: float = 0.0) -> None: ...
    pass
class FileTypes():
    """
    The type of files that the `FileSourceStage` can read and `WriteToFileStage` can write. Use 'auto' to determine from the file extension.

    Members:

      Auto

      JSON

      CSV

      PARQUET
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
    Auto: morpheus._lib.common.FileTypes # value = <FileTypes.Auto: 0>
    CSV: morpheus._lib.common.FileTypes # value = <FileTypes.CSV: 2>
    JSON: morpheus._lib.common.FileTypes # value = <FileTypes.JSON: 1>
    PARQUET: morpheus._lib.common.FileTypes # value = <FileTypes.PARQUET: 3>
    __members__: dict # value = {'Auto': <FileTypes.Auto: 0>, 'JSON': <FileTypes.JSON: 1>, 'CSV': <FileTypes.CSV: 2>, 'PARQUET': <FileTypes.PARQUET: 3>}
    pass
class FilterSource():
    """
    Enum to indicate which source the FilterDetectionsStage should operate on.

    Members:

      Auto

      TENSOR

      DATAFRAME
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
    Auto: morpheus._lib.common.FilterSource # value = <FilterSource.Auto: 0>
    DATAFRAME: morpheus._lib.common.FilterSource # value = <FilterSource.DATAFRAME: 2>
    TENSOR: morpheus._lib.common.FilterSource # value = <FilterSource.TENSOR: 1>
    __members__: dict # value = {'Auto': <FilterSource.Auto: 0>, 'TENSOR': <FilterSource.TENSOR: 1>, 'DATAFRAME': <FilterSource.DATAFRAME: 2>}
    pass
class HttpEndpoint():
    def __init__(self, py_parse_fn: function, url: str, method: str, include_headers: bool = False) -> None: ...
    pass
class HttpServer():
    def __enter__(self) -> HttpServer: ...
    def __exit__(self, arg0: object, arg1: object, arg2: object) -> None: ...
    def __init__(self, endpoints: typing.List[HttpEndpoint], bind_address: str = '127.0.0.1', port: int = 8080, num_threads: int = 1, max_payload_size: int = 10485760, request_timeout: int = 30) -> None: ...
    def is_running(self) -> bool: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    pass
class IndicatorsFontStyle():
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
    __members__: dict # value = {'bold': <IndicatorsFontStyle.bold: 0>, 'dark': <IndicatorsFontStyle.dark: 1>, 'italic': <IndicatorsFontStyle.italic: 2>, 'underline': <IndicatorsFontStyle.underline: 3>, 'blink': <IndicatorsFontStyle.blink: 4>, 'reverse': <IndicatorsFontStyle.reverse: 5>, 'concealed': <IndicatorsFontStyle.concealed: 6>, 'crossed': <IndicatorsFontStyle.crossed: 7>}
    blink: morpheus._lib.common.IndicatorsFontStyle # value = <IndicatorsFontStyle.blink: 4>
    bold: morpheus._lib.common.IndicatorsFontStyle # value = <IndicatorsFontStyle.bold: 0>
    concealed: morpheus._lib.common.IndicatorsFontStyle # value = <IndicatorsFontStyle.concealed: 6>
    crossed: morpheus._lib.common.IndicatorsFontStyle # value = <IndicatorsFontStyle.crossed: 7>
    dark: morpheus._lib.common.IndicatorsFontStyle # value = <IndicatorsFontStyle.dark: 1>
    italic: morpheus._lib.common.IndicatorsFontStyle # value = <IndicatorsFontStyle.italic: 2>
    reverse: morpheus._lib.common.IndicatorsFontStyle # value = <IndicatorsFontStyle.reverse: 5>
    underline: morpheus._lib.common.IndicatorsFontStyle # value = <IndicatorsFontStyle.underline: 3>
    pass
class IndicatorsTextColor():
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
    __members__: dict # value = {'grey': <IndicatorsTextColor.grey: 0>, 'red': <IndicatorsTextColor.red: 1>, 'green': <IndicatorsTextColor.green: 2>, 'yellow': <IndicatorsTextColor.yellow: 3>, 'blue': <IndicatorsTextColor.blue: 4>, 'magenta': <IndicatorsTextColor.magenta: 5>, 'cyan': <IndicatorsTextColor.cyan: 6>, 'white': <IndicatorsTextColor.white: 7>, 'unspecified': <IndicatorsTextColor.unspecified: 8>}
    blue: morpheus._lib.common.IndicatorsTextColor # value = <IndicatorsTextColor.blue: 4>
    cyan: morpheus._lib.common.IndicatorsTextColor # value = <IndicatorsTextColor.cyan: 6>
    green: morpheus._lib.common.IndicatorsTextColor # value = <IndicatorsTextColor.green: 2>
    grey: morpheus._lib.common.IndicatorsTextColor # value = <IndicatorsTextColor.grey: 0>
    magenta: morpheus._lib.common.IndicatorsTextColor # value = <IndicatorsTextColor.magenta: 5>
    red: morpheus._lib.common.IndicatorsTextColor # value = <IndicatorsTextColor.red: 1>
    unspecified: morpheus._lib.common.IndicatorsTextColor # value = <IndicatorsTextColor.unspecified: 8>
    white: morpheus._lib.common.IndicatorsTextColor # value = <IndicatorsTextColor.white: 7>
    yellow: morpheus._lib.common.IndicatorsTextColor # value = <IndicatorsTextColor.yellow: 3>
    pass
class Tensor():
    @staticmethod
    def from_cupy(arg0: object) -> Tensor: ...
    def to_cupy(self) -> object: ...
    @property
    def __cuda_array_interface__(self) -> dict:
        """
        :type: dict
        """
    pass
class TypeId():
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
    BOOL8: morpheus._lib.common.TypeId # value = <TypeId.BOOL8: 11>
    EMPTY: morpheus._lib.common.TypeId # value = <TypeId.EMPTY: 0>
    FLOAT32: morpheus._lib.common.TypeId # value = <TypeId.FLOAT32: 9>
    FLOAT64: morpheus._lib.common.TypeId # value = <TypeId.FLOAT64: 10>
    INT16: morpheus._lib.common.TypeId # value = <TypeId.INT16: 2>
    INT32: morpheus._lib.common.TypeId # value = <TypeId.INT32: 3>
    INT64: morpheus._lib.common.TypeId # value = <TypeId.INT64: 4>
    INT8: morpheus._lib.common.TypeId # value = <TypeId.INT8: 1>
    STRING: morpheus._lib.common.TypeId # value = <TypeId.STRING: 12>
    UINT16: morpheus._lib.common.TypeId # value = <TypeId.UINT16: 6>
    UINT32: morpheus._lib.common.TypeId # value = <TypeId.UINT32: 7>
    UINT64: morpheus._lib.common.TypeId # value = <TypeId.UINT64: 8>
    UINT8: morpheus._lib.common.TypeId # value = <TypeId.UINT8: 5>
    __members__: dict # value = {'EMPTY': <TypeId.EMPTY: 0>, 'INT8': <TypeId.INT8: 1>, 'INT16': <TypeId.INT16: 2>, 'INT32': <TypeId.INT32: 3>, 'INT64': <TypeId.INT64: 4>, 'UINT8': <TypeId.UINT8: 5>, 'UINT16': <TypeId.UINT16: 6>, 'UINT32': <TypeId.UINT32: 7>, 'UINT64': <TypeId.UINT64: 8>, 'FLOAT32': <TypeId.FLOAT32: 9>, 'FLOAT64': <TypeId.FLOAT64: 10>, 'BOOL8': <TypeId.BOOL8: 11>, 'STRING': <TypeId.STRING: 12>}
    pass
@typing.overload
def determine_file_type(filename: os.PathLike) -> FileTypes:
    pass
@typing.overload
def determine_file_type(filename: str) -> FileTypes:
    pass
def read_file_to_df(filename: str, file_type: FileTypes = FileTypes.Auto) -> object:
    pass
def typeid_is_fully_supported(arg0: TypeId) -> bool:
    pass
def typeid_to_numpy_str(arg0: TypeId) -> str:
    pass
def write_df_to_file(df: object, filename: str, file_type: FileTypes = FileTypes.Auto, **kwargs) -> None:
    pass
__version__ = '25.2.0'
