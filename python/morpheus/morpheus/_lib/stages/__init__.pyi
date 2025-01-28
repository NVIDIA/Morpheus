"""
        -----------------------
        .. currentmodule:: morpheus.stages
        .. autosummary::
           :toctree: _generate

        """
from __future__ import annotations
import morpheus._lib.stages
import typing
from morpheus._lib.common import FilterSource
import morpheus._lib.common
import morpheus._lib.messages
import mrc.core.coro
import mrc.core.segment
import os

__all__ = [
    "AddClassificationsStage",
    "AddScoresStage",
    "DeserializeStage",
    "FileSourceStage",
    "FilterDetectionsStage",
    "FilterSource",
    "HttpServerControlMessageSourceStage",
    "HttpServerMessageMetaSourceStage",
    "InferenceClientStage",
    "KafkaSourceStage",
    "MonitorControlMessageStage",
    "MonitorMessageMetaStage",
    "PreallocateControlMessageStage",
    "PreallocateMessageMetaStage",
    "PreprocessFILStage",
    "PreprocessNLPStage",
    "SerializeStage",
    "WriteToFileStage"
]


class AddClassificationsStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, idx2label: typing.Dict[int, str], threshold: float) -> None: ...
    pass
class AddScoresStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, idx2label: typing.Dict[int, str]) -> None: ...
    pass
class DeserializeStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, batch_size: int, ensure_sliceable_index: bool = True, task_type: object = None, task_payload: object = None) -> None: ...
    pass
class FileSourceStage(mrc.core.segment.SegmentObject):
    @typing.overload
    def __init__(self, builder: mrc.core.segment.Builder, name: str, filename: os.PathLike, repeat: int, filter_null: bool, filter_null_columns: typing.List[str], parser_kwargs: dict) -> None: ...
    @typing.overload
    def __init__(self, builder: mrc.core.segment.Builder, name: str, filename: str, repeat: int, filter_null: bool, filter_null_columns: typing.List[str], parser_kwargs: dict) -> None: ...
    pass
class FilterDetectionsStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, threshold: float, copy: bool, filter_source: morpheus._lib.common.FilterSource, field_name: str = 'probs') -> None: ...
    pass
class HttpServerControlMessageSourceStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, bind_address: str = '127.0.0.1', port: int = 8080, endpoint: str = '/message', live_endpoint: str = '/live', ready_endpoint: str = '/ready', method: str = 'POST', live_method: str = 'GET', ready_method: str = 'GET', accept_status: int = 201, sleep_time: float = 0.10000000149011612, queue_timeout: int = 5, max_queue_size: int = 1024, num_server_threads: int = 1, max_payload_size: int = 10485760, request_timeout: int = 30, lines: bool = False, stop_after: int = 0, task_type: object = None, task_payload: object = None) -> None: ...
    pass
class HttpServerMessageMetaSourceStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, bind_address: str = '127.0.0.1', port: int = 8080, endpoint: str = '/message', live_endpoint: str = '/live', ready_endpoint: str = '/ready', method: str = 'POST', live_method: str = 'GET', ready_method: str = 'GET', accept_status: int = 201, sleep_time: float = 0.10000000149011612, queue_timeout: int = 5, max_queue_size: int = 1024, num_server_threads: int = 1, max_payload_size: int = 10485760, request_timeout: int = 30, lines: bool = False, stop_after: int = 0) -> None: ...
    pass
class InferenceClientStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, server_url: str, model_name: str, needs_logits: bool, force_convert_inputs: bool, input_mapping: typing.Dict[str, str] = {}, output_mapping: typing.Dict[str, str] = {}) -> None: ...
    pass
class KafkaSourceStage(mrc.core.segment.SegmentObject):
    @typing.overload
    def __init__(self, builder: mrc.core.segment.Builder, name: str, max_batch_size: int, topic: str, batch_timeout_ms: int, config: typing.Dict[str, str], disable_commits: bool = False, disable_pre_filtering: bool = False, stop_after: int = 0, async_commits: bool = True, oauth_callback: typing.Optional[function] = None) -> None: ...
    @typing.overload
    def __init__(self, builder: mrc.core.segment.Builder, name: str, max_batch_size: int, topics: typing.List[str], batch_timeout_ms: int, config: typing.Dict[str, str], disable_commits: bool = False, disable_pre_filtering: bool = False, stop_after: int = 0, async_commits: bool = True, oauth_callback: typing.Optional[function] = None) -> None: ...
    pass
class MonitorControlMessageStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, description: str, unit: str = 'messages', text_color: morpheus._lib.common.IndicatorsTextColor = IndicatorsTextColor.cyan, font_style: morpheus._lib.common.IndicatorsFontStyle = IndicatorsFontStyle.bold, determine_count_fn: typing.Optional[typing.Callable[[morpheus._lib.messages.ControlMessage], int]] = None) -> None: ...
    pass
class MonitorMessageMetaStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, description: str, unit: str = 'messages', text_color: morpheus._lib.common.IndicatorsTextColor = IndicatorsTextColor.cyan, font_style: morpheus._lib.common.IndicatorsFontStyle = IndicatorsFontStyle.bold, determine_count_fn: typing.Optional[typing.Callable[[morpheus._lib.messages.MessageMeta], int]] = None) -> None: ...
    pass
class PreallocateControlMessageStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, needed_columns: typing.List[typing.Tuple[str, morpheus._lib.common.TypeId]]) -> None: ...
    pass
class PreallocateMessageMetaStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, needed_columns: typing.List[typing.Tuple[str, morpheus._lib.common.TypeId]]) -> None: ...
    pass
class PreprocessFILStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, features: typing.List[str]) -> None: ...
    pass
class PreprocessNLPStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, vocab_hash_file: str, sequence_length: int, truncation: bool, do_lower_case: bool, add_special_token: bool, stride: int, column: str) -> None: ...
    pass
class SerializeStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, include: typing.List[str], exclude: typing.List[str], fixed_columns: bool = True) -> None: ...
    pass
class WriteToFileStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, filename: str, mode: str = 'w', file_type: morpheus._lib.common.FileTypes = FileTypes.Auto, include_index_col: bool = True, flush: bool = False) -> None: ...
    pass
__version__ = '25.2.0'
