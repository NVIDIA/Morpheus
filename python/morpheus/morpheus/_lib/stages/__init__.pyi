"""

        -----------------------
        .. currentmodule:: morpheus._lib.stages
        .. autosummary::
           :toctree: _generate

        
"""
from __future__ import annotations
import morpheus._lib.common
from morpheus._lib.common import FileTypes
from morpheus._lib.common import IndicatorsFontStyle
from morpheus._lib.common import IndicatorsTextColor
import morpheus._lib.messages
import mrc.core.segment
import os
import typing
__all__ = ['AddClassificationsStage', 'AddScoresStage', 'DeserializeStage', 'FileSourceStage', 'FileTypes', 'FilterDetectionsStage', 'HttpServerControlMessageSourceStage', 'HttpServerMessageMetaSourceStage', 'IndicatorsFontStyle', 'IndicatorsTextColor', 'InferenceClientStage', 'KafkaSourceStage', 'MonitorControlMessageStage', 'MonitorMessageMetaStage', 'PreallocateControlMessageStage', 'PreallocateMessageMetaStage', 'PreprocessFILStage', 'PreprocessNLPStage', 'SerializeStage', 'WriteToFileStage']
class AddClassificationsStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, idx2label: dict[int, str], threshold: float) -> None:
        ...
class AddScoresStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, idx2label: dict[int, str]) -> None:
        ...
class DeserializeStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, batch_size: int, ensure_sliceable_index: bool = True, task_type: typing.Any = None, task_payload: typing.Any = None) -> None:
        ...
class FileSourceStage(mrc.core.segment.SegmentObject):
    @typing.overload
    def __init__(self, builder: mrc.core.segment.Builder, name: str, filename: str, repeat: int, filter_null: bool, filter_null_columns: list[str], parser_kwargs: dict) -> None:
        ...
    @typing.overload
    def __init__(self, builder: mrc.core.segment.Builder, name: str, filename: os.PathLike, repeat: int, filter_null: bool, filter_null_columns: list[str], parser_kwargs: dict) -> None:
        ...
class FilterDetectionsStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, threshold: float, copy: bool, filter_source: morpheus._lib.common.FilterSource, field_name: str = 'probs') -> None:
        ...
class HttpServerControlMessageSourceStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, bind_address: str = '127.0.0.1', port: int = 8080, endpoint: str = '/message', live_endpoint: str = '/live', ready_endpoint: str = '/ready', method: str = 'POST', live_method: str = 'GET', ready_method: str = 'GET', accept_status: int = 201, sleep_time: float = 0.10000000149011612, queue_timeout: int = 5, max_queue_size: int = 1024, num_server_threads: int = 1, max_payload_size: int = 10485760, request_timeout: int = 30, lines: bool = False, stop_after: int = 0, task_type: typing.Any = None, task_payload: typing.Any = None) -> None:
        ...
class HttpServerMessageMetaSourceStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, bind_address: str = '127.0.0.1', port: int = 8080, endpoint: str = '/message', live_endpoint: str = '/live', ready_endpoint: str = '/ready', method: str = 'POST', live_method: str = 'GET', ready_method: str = 'GET', accept_status: int = 201, sleep_time: float = 0.10000000149011612, queue_timeout: int = 5, max_queue_size: int = 1024, num_server_threads: int = 1, max_payload_size: int = 10485760, request_timeout: int = 30, lines: bool = False, stop_after: int = 0) -> None:
        ...
class InferenceClientStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, server_url: str, model_name: str, needs_logits: bool, force_convert_inputs: bool, input_mapping: dict[str, str] = {}, output_mapping: dict[str, str] = {}) -> None:
        ...
class KafkaSourceStage(mrc.core.segment.SegmentObject):
    @typing.overload
    def __init__(self, builder: mrc.core.segment.Builder, name: str, max_batch_size: int, topic: str, batch_timeout_ms: int, config: dict[str, str], disable_commits: bool = False, disable_pre_filtering: bool = False, stop_after: int = 0, async_commits: bool = True, oauth_callback: typing.Callable | None = None) -> None:
        ...
    @typing.overload
    def __init__(self, builder: mrc.core.segment.Builder, name: str, max_batch_size: int, topics: list[str], batch_timeout_ms: int, config: dict[str, str], disable_commits: bool = False, disable_pre_filtering: bool = False, stop_after: int = 0, async_commits: bool = True, oauth_callback: typing.Callable | None = None) -> None:
        ...
class MonitorControlMessageStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, description: str, unit: str = 'messages', text_color: morpheus._lib.common.IndicatorsTextColor = IndicatorsTextColor.cyan, font_style: morpheus._lib.common.IndicatorsFontStyle = IndicatorsFontStyle.bold, determine_count_fn: typing.Callable[[morpheus._lib.messages.ControlMessage], int] | None = None) -> None:
        ...
class MonitorMessageMetaStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, description: str, unit: str = 'messages', text_color: morpheus._lib.common.IndicatorsTextColor = IndicatorsTextColor.cyan, font_style: morpheus._lib.common.IndicatorsFontStyle = IndicatorsFontStyle.bold, determine_count_fn: typing.Callable[[morpheus._lib.messages.MessageMeta], int] | None = None) -> None:
        ...
class PreallocateControlMessageStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, needed_columns: list[tuple[str, morpheus._lib.common.TypeId]]) -> None:
        ...
class PreallocateMessageMetaStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, needed_columns: list[tuple[str, morpheus._lib.common.TypeId]]) -> None:
        ...
class PreprocessFILStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, features: list[str]) -> None:
        ...
class PreprocessNLPStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, vocab_hash_file: str, sequence_length: int, truncation: bool, do_lower_case: bool, add_special_token: bool, stride: int, column: str) -> None:
        ...
class SerializeStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, include: list[str], exclude: list[str], fixed_columns: bool = True) -> None:
        ...
class WriteToFileStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, filename: str, mode: str = 'w', file_type: morpheus._lib.common.FileTypes = FileTypes.Auto, include_index_col: bool = True, flush: bool = False) -> None:
        ...
__version__: str = '25.2.0'
