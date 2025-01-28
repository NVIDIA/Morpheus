# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

import mrc
import pytest

from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.configurable_output_source import ConfigurableOutputSource
from morpheus.pipeline.configurable_output_source import SupportedMessageTypes
from morpheus.pipeline.stage_schema import StageSchema


class ConfigurableOutputSourceImpl(ConfigurableOutputSource):
    """
    Concrete implementation of ConfigurableOutputSource for testing purposes.
    """

    @property
    def name(self) -> str:
        return "from-unit-test"

    def supports_cpp_node(self) -> bool:
        return False

    def _build_source(self, _: mrc.Builder) -> mrc.SegmentObject:
        raise NotImplementedError("Not implemented in test")


@pytest.mark.parametrize("message_type", SupportedMessageTypes)
def test_compute_schema(config: Config, message_type: SupportedMessageTypes):
    source = ConfigurableOutputSourceImpl(config=config, message_type=message_type)

    if message_type == SupportedMessageTypes.MESSAGE_META:
        expected_message_class = MessageMeta
    else:
        expected_message_class = ControlMessage

    schema = StageSchema(source)
    source.compute_schema(schema)

    assert len(schema.output_schemas) == 1

    port_schema = schema.output_schemas[0]
    assert port_schema.get_type() is expected_message_class


def test_constructor_error_task_with_message_meta(config: Config):
    with pytest.raises(ValueError):
        ConfigurableOutputSourceImpl(config=config,
                                     message_type=SupportedMessageTypes.MESSAGE_META,
                                     task_type="test",
                                     task_payload={"why": "setting task only valid for ControlMessage output"})


def test_constructor_error_task_type_without_task_payload(config: Config):
    with pytest.raises(ValueError):
        ConfigurableOutputSourceImpl(config=config,
                                     message_type=SupportedMessageTypes.CONTROL_MESSAGE,
                                     task_type="setting task_type requires setting task_payload")


def test_constructor_error_task_payload_without_task_type(config: Config):
    with pytest.raises(ValueError):
        ConfigurableOutputSourceImpl(config=config,
                                     message_type=SupportedMessageTypes.MESSAGE_META,
                                     task_payload={"why": "setting task_payload requires setting task_type"})


def test_constructor_error_invalid_type(config: Config):
    with pytest.raises(ValueError):
        ConfigurableOutputSourceImpl(config=config, message_type="invalid message type")
