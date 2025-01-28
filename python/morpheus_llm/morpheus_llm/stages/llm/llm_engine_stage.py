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

import functools
import logging
import types
import typing

import mrc
from mrc.core import operators as ops

from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.messages import ControlMessage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus_llm.llm import LLMEngine

logger = logging.getLogger(__name__)


class LLMEngineStage(PassThruTypeMixin, GpuAndCpuMixin, SinglePortStage):
    """
    Stage for executing an LLM engine within a Morpheus pipeline.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    engine : `morpheus_llm.llm.LLMEngine`
        LLM engine instance to execute.
   """

    def __init__(self, c: Config, *, engine: LLMEngine):
        super().__init__(c)

        self._engine = engine

    @property
    def name(self) -> str:
        """Return the name of the stage"""
        return "llm-engine"

    def accepted_types(self) -> tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        tuple(`ControlMessage`, )
            Accepted input types.

        """
        return (ControlMessage, )

    def supports_cpp_node(self) -> bool:
        """Indicates whether this stage supports a C++ node."""
        return True

    def _store_payload(self, message: ControlMessage) -> ControlMessage:
        """
        Store the MessageMeta in the ControlMessage's metadata.

        In CPU-only allows the ControlMessage to hold an instance of a Python MessageMeta containing a pandas DataFrame.
        """
        message.set_metadata("llm_message_meta", message.payload())
        return message

    def _copy_tasks_and_metadata(self,
                                 src: ControlMessage,
                                 dst: ControlMessage,
                                 metadata: dict[str, typing.Any] = None):
        if metadata is None:
            metadata = src.get_metadata()

        for (key, value) in metadata.items():
            dst.set_metadata(key, value)

        tasks = src.get_tasks()
        for (task, task_value) in tasks.items():
            for tv in task_value:
                dst.add_task(task, tv)

        timestamps = src.get_timestamps()
        for (ts_key, ts) in timestamps.items():
            dst.set_timestamp(key=ts_key, timestamp=ts)

    def _cast_to_cpp_control_message(self, py_message: ControlMessage, *,
                                     cpp_messages_lib: types.ModuleType) -> ControlMessage:
        """
        LLMEngineStage does not contain a Python implementation, however it is capable of running in cpu-only mode.
        This method is needed to create an instance of a C++ ControlMessage.

        This is different than casting from the Python bindings for the C++ ControlMessage to a C++ ControlMessage.
        """
        cpp_message = cpp_messages_lib.ControlMessage()
        self._copy_tasks_and_metadata(py_message, cpp_message)

        return cpp_message

    def _restore_payload(self, message: ControlMessage) -> ControlMessage:
        """
        Pop llm_message_meta from the metadata and set it as the payload.

        In CPU-only mode this has the effect of converting the C++ ControlMessage back to a Python ControlMessage.
        """
        metadata = message.get_metadata()
        message_meta = metadata.pop("llm_message_meta")

        out_message = ControlMessage()
        out_message.payload(message_meta)

        self._copy_tasks_and_metadata(message, out_message, metadata=metadata)

        return out_message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        import morpheus_llm._lib.llm as _llm

        store_payload_node = builder.make_node(f"{self.unique_name}-store-payload", ops.map(self._store_payload))
        builder.make_edge(input_node, store_payload_node)

        node = _llm.LLMEngineStage(builder, self.unique_name, self._engine)
        node.launch_options.pe_count = 1

        if self._config.execution_mode == ExecutionMode.CPU:
            import morpheus._lib.messages as _messages
            cast_to_cpp_fn = functools.partial(self._cast_to_cpp_control_message, cpp_messages_lib=_messages)
            cast_to_cpp_node = builder.make_node(f"{self.unique_name}-pre-msg-cast", ops.map(cast_to_cpp_fn))
            builder.make_edge(store_payload_node, cast_to_cpp_node)
            builder.make_edge(cast_to_cpp_node, node)

        else:
            builder.make_edge(store_payload_node, node)

        restore_payload_node = builder.make_node(f"{self.unique_name}-restore-payload", ops.map(self._restore_payload))
        builder.make_edge(node, restore_payload_node)

        return restore_payload_node
